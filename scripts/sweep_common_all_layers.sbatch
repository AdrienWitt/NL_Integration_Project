#!/bin/bash
#SBATCH --job-name=sweep_common
#SBATCH --output=logs/swc_%a_out
#SBATCH --error=logs/swc_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:24G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-35
#
# ===========================================================================
# Every stored layer of every model, on the common stories, per subject.
# ===========================================================================
#
# One task per (store, subject); 4 x 9 = 36. Each writes ONE sweep.csv holding
# one row per layer, which is the format to ask for when the question is "how
# does this model behave across depth" — `run_encoding` writes a directory per
# band instead, so covering 24 layers there would mean 24 directories to
# reassemble by hand.
#
# Common stories only
# -------------------
# common_stories_all9.json puts all nine subjects on the same 24 training
# stories, so scores can be averaged across subjects. The per-subject lists
# (26-84 stories) make a group mean part prosody and part "UTS01 heard three
# times more data" — see scripts/make_common_stories.py for why the shipped
# common_stories_25.json cannot be used for this.
#
# It also makes the whole thing cheap: 24 stories is ~8.7k training TRs against
# 29k, and kernel ridge is cubic, so roughly 40x less per fit. That is what
# makes sweeping every layer affordable rather than a considered subset.
#
# Both evaluations, and they are not interchangeable
# --------------------------------------------------
# --eval cv     cross-validated inside the 24 training stories. The honest
#               basis for CHOOSING a layer: the held-out story is never seen,
#               so picking the winner here costs nothing.
# --eval holdout  scored on wheretheressmoke, the repeated story. The number to
#               REPORT for a layer already chosen. Picking the best row from a
#               holdout sweep is selection on the test set, and the winner's
#               score is then optimistic — read that profile as a shape, and
#               take the selection from the cv run.
#
# Both are written, to different output roots, because the pair is what makes
# the claim defensible: choose on cv, report on holdout, and say so.

set -e
mkdir -p logs

# Log to node-local disk, copy once at exit.
#
# BeeGFS on this cluster intermittently returns Errno 121 (Remote I/O error) on
# the steady trickle of writes a long job makes -- on /home at 97% full, and on
# scratch too. Under `set -e` a failed `echo` is fatal, so the first run of this
# array lost 18 of 36 tasks to a banner line, each of them mid-sweep with the
# holdout half never started. Nothing was wrong with the computation.
#
# Redirecting the whole script to $TMPDIR removes every incremental write to the
# shared filesystem; the copy at exit is one write, and it is allowed to fail
# without taking the job with it. Results still land on BeeGFS, but those are a
# handful of large writes rather than thousands of small ones, and they have
# been reliable throughout.
LOG="${TMPDIR:-/tmp}/swc_${SLURM_ARRAY_TASK_ID}.log"
trap 'cp -f "$LOG" "logs/swc_${SLURM_ARRAY_TASK_ID}_full.log" 2>/dev/null || true' EXIT
exec > "$LOG" 2>&1

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
STORES=(perlayer_base_robust perlayer_ft_robust
        perlayer_base_emotion perlayer_ft_emotion)

# Every stored layer, plus a few averaged ranges. The ranges are here because
# averaging neighbouring layers sometimes beats any single one, and the sweep
# is the only place that can be tested honestly; `build_band --mode mean` is
# the same operation, so a winning range is directly materialisable.
CONFIGS=("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 12-17 15-18 18-23"
         "12 13 14 15 16 17 18 19 20 21 22 23 12-17 18-23"
         "0 1 2 3 4 5 6 7 8 9 10 11 6-11 9-11"
         "6 7 8 9 10 11 6-11 9-11")

# Store-major: tasks 0-8 are base_robust across all nine subjects, so a partial
# queue still yields one complete model rather than nine partial ones.
NSUBJ=${#SUBJECTS[@]}
IDX=$((SLURM_ARRAY_TASK_ID / NSUBJ))
SUBJ=${SUBJECTS[$((SLURM_ARRAY_TASK_ID % NSUBJ))]}
STORE=${STORES[$IDX]}
CFG=${CONFIGS[$IDX]}

RESP="data/ds003020/derivative/preprocessed_data/$SUBJ"
if [ ! -d "$RESP" ]; then
    echo "ERROR: no responses for $SUBJ at $RESP"
    exit 1
fi

echo "=== task $SLURM_ARRAY_TASK_ID : $SUBJ / $STORE ==="
echo "    configs: $CFG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_eval () {   # eval mode
    echo "########## $SUBJ / $STORE / --eval $1 : t+$((SECONDS / 60)) min"
    python -m encoding.run_prosody_sweep \
        --subjects "$SUBJ" \
        --source "$STORE" \
        --configs "$CFG" \
        --stories-json common_stories_all9.json \
        --baseline-features opensmile \
        --tag "$SUBJ" \
        --eval "$1" \
        --n-splits 5 \
        --min-ev 0.1 \
        --himalaya-backend torch_cuda
}

run_eval cv
run_eval holdout

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
