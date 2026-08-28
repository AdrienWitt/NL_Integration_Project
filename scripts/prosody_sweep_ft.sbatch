#!/bin/bash
#SBATCH --job-name=sweep_ft
#SBATCH --output=sweep_ft_%a_out
#SBATCH --error=sweep_ft_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
# --mem=48G is generous. Responses are masked to the explainable-variance
# voxels as each story is read, so the response matrix is ~0.4 GB rather than
# the 18 GB it would be at all 81k voxels; measured peak is nearer 12 GB.
# The binding constraint is GPU memory: himalaya's torch_cuda backend holds an
# (n_TRs, n_TRs) kernel, ~0.7 GB at 40 stories. Drop VramPerGpu to 11G to also
# match the RTX 2080 Ti / P100 nodes if the A100 queue is long.

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
MAXSTORIES=40

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
run_sweep perlayer_ft_robust    "12 14 16 18 20 22 23"
run_sweep perlayer_ft_emotion   "6 7 8 9 10 11"

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
