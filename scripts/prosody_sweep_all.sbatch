#!/bin/bash
#SBATCH --job-name=sweep_all
#SBATCH --output=logs/sweep_all_%a_out
#SBATCH --error=logs/sweep_all_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:24G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-35
#
# ===========================================================================
# The full prosodic layer sweep: 9 subjects x 4 stores, one GPU each.
# ===========================================================================
#
# How the parallelism works, and why it is shaped this way
# --------------------------------------------------------
# The unit of independent work is (subject, store). Nothing is shared between
# two such pairs — different features, different responses, different output
# directory — so all 36 can run at once on 36 different cards, anywhere in the
# cluster. That is the whole design: SLURM does the scheduling, the script does
# nothing clever, and no two tasks ever need to be on the same node.
#
# Three things follow, and they are the ones that actually decide throughput:
#
# 1. ONE GPU PER TASK, NOT SEVERAL. himalaya's torch_cuda backend sends every
#    array to plain "cuda" -- device 0 -- with no sharding and no
#    DataParallel. --gres=gpu:4 would reserve four cards and use one. Worse,
#    it would demand four *co-located* cards, so the job waits for a whole
#    node while 36 single-GPU tasks would already be running on scraps.
#    More parallelism comes from more tasks, never from more cards per task.
#
# 2. VramPerGpu:24G, NOT 80G. This is the single biggest throughput knob in
#    the file. Measured pool in shared-gpu:
#
#        >= 80G  ~9 cards    (gpu027/029/045 + the H100s)
#        >= 40G  ~37 cards
#        >= 24G  ~129 cards
#
#    A 40-story fit needs a (13329 x 13329) float32 kernel = 0.7 GB plus
#    solver working copies -- comfortably inside 24 GB. Asking for 80 GB would
#    cut the reachable pool by 14x to buy headroom nothing uses. Raise it only
#    if you raise MAXSTORIES towards all 83, where the kernel reaches 3.4 GB
#    and the working set 20-35 GB.
#
# 3. STORE-MAJOR TASK ORDER. Task id maps to (store, subject) with the store
#    varying slowest, so tasks 0-8 are ft_robust across all nine subjects,
#    9-17 ft_emotion, and only then the frozen baselines. Under a %N throttle
#    the queue therefore answers "did fine-tuning work" across the whole
#    cohort before it spends a single card on a control. Reverse the two lines
#    marked below if you would rather have complete results per subject.
#
# No %N throttle: all 36 tasks are released at once. GPUs on this partition are
# usually plentiful and 24G reaches ~129 cards, so 36 tasks land immediately
# and the whole sweep is one task's wall clock, not thirty-six. Add %N back
# only if you are told you are crowding the partition.
#
# --time=02:00:00 is deliberately tight. With diagonalize_method="eigh" a
# configuration costs seconds, so a task is minutes; a short request also
# schedules ahead of long ones. If a task ever hits this wall, something is
# wrong with the fit, not with the budget.

set -e

# logs/ must exist BEFORE submission: SLURM opens the --output file itself,
# before this script runs, and a missing directory fails the task silently
# from the user's point of view. The mkdir below only covers a later rerun.
#   mkdir -p logs && sbatch scripts/prosody_sweep_all.sbatch
mkdir -p logs

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
STORES=(perlayer_ft_robust perlayer_ft_emotion
        perlayer_base_robust perlayer_base_emotion)
CONFIGS=("12 14 16 18 20 22 23"
         "6 7 8 9 10 11"
         "0 3 6 9 12 15 18 21 23"
         "0 3 6 9 11")

# Store-major: store varies slowest, subject fastest. Swap the two divisors to
# make it subject-major instead.
NSUBJ=${#SUBJECTS[@]}
STORE_ID=$((SLURM_ARRAY_TASK_ID / NSUBJ))
SUBJ_ID=$((SLURM_ARRAY_TASK_ID % NSUBJ))

SUBJ=${SUBJECTS[$SUBJ_ID]}
STORE=${STORES[$STORE_ID]}
CFG=${CONFIGS[$STORE_ID]}

# The fine-tuned stores start at their freeze boundary: below it the weights
# are byte-identical to the pretrained base, so those layers ARE the base
# profile and sweeping them in both places would measure the same thing twice.
# robust is 24 layers frozen to 12; emotion is a 12-layer pruned backbone
# frozen to 6, so its "layer 11" is the top of the stack, not mid-network.

# Training stories per fit. The sweep needs the ranking of layers, not a
# publishable number, and kernel ridge is cubic in TRs -- 40 stories costs a
# fraction of all 83. Subjects who heard fewer simply use what they have:
# only UTS01-03 heard all 84, the rest heard 26-54 (see
# data/derivative/common_stories_*.json).
MAXSTORIES=${MAXSTORIES:-40}

# eGeMAPS is scored inside every task, on that task's own folds and mask. It is
# refitted 4x per subject rather than shared, which is deliberate: it costs
# about one configuration and it makes each store's sweep.csv readable on its
# own, with its reference line already in it. Set BASELINE= to switch it off.
BASELINE=${BASELINE-opensmile}

RESP="data/ds003020/derivative/preprocessed_data/$SUBJ"
if [ ! -d "$RESP" ]; then
    echo "ERROR: no responses for $SUBJ at $RESP"
    exit 1
fi

echo "=== task $SLURM_ARRAY_TASK_ID : $SUBJ / $STORE ==="
echo "    configs   : $CFG"
echo "    stories   : $(ls "$RESP"/*.hf5 2>/dev/null | wc -l) available, cap $MAXSTORIES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BASELINE_ARG=""
if [ -n "$BASELINE" ]; then
    BASELINE_ARG="--baseline-features $BASELINE"
fi

# --tag "$SUBJ" is not cosmetic. The sweep summary lands at
# results/encoding/prosody_sweep/<eval>/<source>/sweep.csv, a path with no
# subject in it, so the nine tasks sharing a store would race for one file and
# only the last to finish would survive.
python -m encoding.run_prosody_sweep \
    --subjects "$SUBJ" \
    --source "$STORE" \
    --configs "$CFG" \
    --tag "$SUBJ" \
    --max-stories "$MAXSTORIES" \
    $BASELINE_ARG \
    --n-splits 5 \
    --eval cv \
    --min-ev 0.1 \
    --himalaya-backend torch_cuda

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
