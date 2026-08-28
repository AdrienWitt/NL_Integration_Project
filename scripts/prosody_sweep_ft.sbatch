#!/bin/bash
#SBATCH --job-name=sweep_ft
#SBATCH --output=sweep_ft_%a_out
#SBATCH --error=sweep_ft_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:80G
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-0
#
# ===========================================================================
# Stage 1 of the prosodic layer sweep: THE FINE-TUNED MODELS ONLY.
# ===========================================================================
#
# The question this job answers is "did fine-tuning on eGeMAPS produce a
# representation that predicts the brain better than the 88 descriptors it was
# distilled from". That needs exactly two things: the fine-tuned layer profile,
# and opensmile scored on the same folds and the same voxels. Both are here.
#
# The frozen-base profiles are the *second* question (did fine-tuning move
# anything, or was the base already there), and they are commented out at the
# bottom rather than deleted. Uncomment and resubmit once this run has told us
# what a configuration actually costs.
#
# Why not everything at once
# --------------------------
# Nothing has been fitted on cluster hardware yet, so the per-configuration
# cost is unmeasured. Kernel ridge is cubic in training TRs, and the only
# timing we have comes from a saturated 8 GB laptop card, which says nothing
# about an A100. Putting all four stores in one 12 h job risks the wall clock
# killing it mid-way; splitting it means the expensive unknown is measured on
# the run that matters most.
#
# --array=0-0 is one subject on purpose: calibrate, read the elapsed times
# this script prints, then widen to 0-8%3. Nine subjects at an unknown cost is
# nine ways to waste a queue slot.
#
# Sizing
# ------
# --mem=64G is host RAM and is generous. Responses are masked to the
# explainable-variance voxels as each story is read, so the response matrix is
# under 1 GB rather than the 18 GB it would be at all 81k voxels; the peak is
# the held-out story's 10 repeats at full voxel space.
#
# GPU memory is the interesting constraint, because it sets how many training
# stories a fit can hold. himalaya's torch_cuda backend builds an
# (n_TRs, n_TRs) kernel and the solver keeps several working copies of it:
#
#   40 stories ~ 14k TRs -> kernel  0.8 GB, working set well under 20 GB
#   all 83     ~ 29k TRs -> kernel  3.4 GB, working set 20-35 GB
#
# VramPerGpu:80G matches the A100 80GB nodes (gpu027/029/045/053) and the H100
# NVL at 94 GB. Note what it does and does not buy: it removes memory as a
# reason the 40-story sweep would thrash, and it is a precondition for ever
# running all 83 — but not a sufficient one. Kernel ridge is cubic in TRs, so
# 83 stories is ~8x the compute of 40, and the first calibration run suggests
# that does not fit a 12 h slot for thirteen configurations. MAXSTORIES=40 is
# the working default for that reason, not a memory one.
#
# Measured on the first real run (20 stories, UTS01): host MaxRSS 7.7 GB,
# 6,897 training TRs, EV>0.1 in 1,776 of 81,126 voxels.
#
# On GPUs and parallelism
# -----------------------
# himalaya's torch_cuda backend sends every array to plain "cuda" — device 0 —
# with no sharding and no DataParallel. It cannot use two cards. Raising
# --gres to gpu:2 therefore does nothing on its own; a second card would sit
# idle for the whole job.
#
# Two GPUs only help if two *processes* use them, which is what the store split
# further down does when it sees a second card. That halves the wall clock for
# a single subject, and is worth asking for when you are running one subject
# and want the answer sooner.
#
# The better axis is across subjects, via the job array — no code involved, no
# requirement that both cards be on one node, and it scales to nine rather than
# two. %N in --array decides how much of the queue this occupies; there are ~7
# usable A100 80GB cards in shared-gpu, so %3 is polite and %7 is the ceiling
# before you are simply waiting on yourself. Asking for gpu:2 *and* a wide
# array competes with itself for the same small pool.
#
# Drop VramPerGpu to 20G to also match the 40 GB A100s and the RTX cards if the
# 80 GB queue is long; the 40-story sweep fits there.

set -e

# bash, not sh: $SECONDS is a bash builtin and the elapsed-time guard below
# depends on it. Under dash it would expand to empty and the [ -gt ] test would
# abort the job on its first call.

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SUBJECTS="UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09"
SUBJ=$(echo $SUBJECTS | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

# Fail early rather than after loading a feature store: a missing subject means
# the response transfer did not finish, which is worth knowing in second one.
RESP="data/ds003020/derivative/preprocessed_data/$SUBJ"
if [ ! -d "$RESP" ]; then
    echo "ERROR: no responses for $SUBJ at $RESP"
    echo "The rsync from the acquisition machine has not delivered this subject."
    exit 1
fi
echo "=== $SUBJ : $(ls "$RESP"/*.hf5 2>/dev/null | wc -l) response files ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Training stories the sweep sees. A sweep needs the *ranking* of layers, not a
# publishable number, and kernel ridge is cubic in TRs, so 40 stories costs a
# fraction of what all 83 would. Lower this first if the job runs long.
#
# These three are env-overridable so a calibration run needs no edit to the
# file — one configuration on few stories tells you what a fit costs before you
# spend a 12 h slot on thirteen of them:
#
#   sbatch --time=01:00:00 \
#          --export=ALL,MAXSTORIES=20,CFG_FT_ROBUST=20,CFG_FT_EMOTION= \
#          scripts/prosody_sweep_ft.sbatch
#
# An empty config list skips that store entirely. The skip uses `if`, not
# `[ ... ] && run_sweep`: under `set -e` a false test there returns non-zero
# for the whole line and kills the job.
MAXSTORIES=${MAXSTORIES:-40}
CFG_FT_ROBUST=${CFG_FT_ROBUST-"12 14 16 18 20 22 23"}
CFG_FT_EMOTION=${CFG_FT_EMOTION-"6 7 8 9 10 11"}

# Stop starting new blocks once the wall clock is nearly gone. Each run_sweep
# writes its own sweep.csv on completion, so a block that never starts costs
# nothing while a block killed at 11h59 loses everything it computed.
BUDGET=$((11 * 3600))

run_sweep () {   # store, configs
    if [ "$SECONDS" -gt "$BUDGET" ]; then
        echo "########## SKIPPED $1 : $((SECONDS / 60)) min elapsed, over budget"
        return 0
    fi
    echo "########## $SUBJ / $1 : starting at $((SECONDS / 60)) min"
    t0=$SECONDS
    python -m encoding.run_prosody_sweep \
        --subjects "$SUBJ" \
        --source "$1" \
        --configs "$2" \
        --tag "$SUBJ" \
        --max-stories "$MAXSTORIES" \
        --baseline-features opensmile \
        --n-splits 5 \
        --eval cv \
        --min-ev 0.1 \
        --himalaya-backend torch_cuda
    echo "########## $SUBJ / $1 : done in $(((SECONDS - t0) / 60)) min"
}

# --tag "$SUBJ" is not cosmetic. The sweep writes its summary to
# results/encoding/prosody_sweep/<eval>/<source>/sweep.csv, a path that does not
# mention the subject, so nine array tasks would race for one file and the last
# to finish would be the only one left. The per-configuration voxel maps were
# always per-subject; only the summary needed the tag.

# ---------------------------------------------------------------------------
# ACTIVE: fine-tuned stores.
#
# Both start at their freeze boundary. Below it the weights are byte-identical
# to the pretrained base, so those layers ARE the base profile and sweeping
# them here would measure the same thing twice.
#
# robust is 24 layers frozen to 12; emotion is a 12-layer pruned backbone
# frozen to 6, so "layer 11" is its top layer, not a mid-network one.
# ---------------------------------------------------------------------------
# The two stores are independent — separate features, separate output
# directories, and the ~20 s of shared setup (responses, EV mask, folds) is
# recomputed per process at negligible cost. So with a second GPU on the node
# they run side by side instead of one after the other.
#
# himalaya cannot itself use two GPUs (see the note at --gres above), so the
# split has to happen at the process level: one process per card, pinned with
# CUDA_VISIBLE_DEVICES. Their logs are separated too, because two processes
# writing to one stream interleave into something unreadable.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "=== $NGPU GPU(s) visible ==="

if [ "$NGPU" -ge 2 ] && [ -n "$CFG_FT_ROBUST" ] && [ -n "$CFG_FT_EMOTION" ]; then
    echo "=== running both stores concurrently, one per GPU ==="
    ( CUDA_VISIBLE_DEVICES=0 run_sweep perlayer_ft_robust  "$CFG_FT_ROBUST" ) \
        > "sweep_${SUBJ}_ft_robust.log"  2>&1 &
    pid_robust=$!
    ( CUDA_VISIBLE_DEVICES=1 run_sweep perlayer_ft_emotion "$CFG_FT_EMOTION" ) \
        > "sweep_${SUBJ}_ft_emotion.log" 2>&1 &
    pid_emotion=$!

    # Wait on each pid separately and keep both exit codes. A bare `wait`
    # returns 0 whatever the children did, which under `set -e` would let a
    # failed sweep pass for a successful job.
    rc=0
    wait $pid_robust  || rc=$?
    wait $pid_emotion || rc=$?
    cat "sweep_${SUBJ}_ft_robust.log" "sweep_${SUBJ}_ft_emotion.log"
    [ "$rc" -eq 0 ] || exit "$rc"
else
    if [ -n "$CFG_FT_ROBUST" ]; then
        run_sweep perlayer_ft_robust  "$CFG_FT_ROBUST"
    fi
    if [ -n "$CFG_FT_EMOTION" ]; then
        run_sweep perlayer_ft_emotion "$CFG_FT_EMOTION"
    fi
fi

# ---------------------------------------------------------------------------
# STAGE 2 — the frozen baselines. Uncomment and resubmit.
#
# These answer "did fine-tuning move anything". Only worth the queue time once
# stage 1 shows the fine-tuned profile, because the interesting comparison is
# shape against shape, and the emotion pair is the clean one: both stores are
# extracted at matched depths, while the robust pair is not.
# ---------------------------------------------------------------------------
# run_sweep perlayer_base_robust  "0 3 6 9 12 15 18 21 23"
# run_sweep perlayer_base_emotion "0 3 6 9 11"

# ---------------------------------------------------------------------------
# STAGE 3 — refine around whatever peaked, and test whether averaging a band
# beats the single best layer. Edit the ranges to bracket the stage-1 peak;
# the values below are a placeholder assuming the top of each stack wins.
# ---------------------------------------------------------------------------
# run_sweep perlayer_ft_robust    "19 20 21 18-23 20-23 12-23"
# run_sweep perlayer_ft_emotion   "9 10 11 6-11 9-11"

# ---------------------------------------------------------------------------
# STAGE 4 — the stricter reading of "prosody": the audio band's split score
# with the semantic band held as a covariate, so what is left is what the
# representation explains *beyond* words.
#
# Deliberately not run now. The semantic band is still a moving target — GPT-2
# layer choice and context weighting are both open — so conditioning on a
# provisional version would date these numbers the moment it changes. Layer
# rankings are usually stable between the two readings; absolute values are not.
# ---------------------------------------------------------------------------
# run_sweep_withtext () { ... add --with-text --text-features gpt2_mean ... }

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
