#!/bin/bash
#SBATCH --job-name=enc_wr
#SBATCH --output=logs/encwr_%a_out
#SBATCH --error=logs/encwr_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-8
#
# ===========================================================================
# Is the long-context text signal in Heschl's gyrus just word rate?
# ===========================================================================
#
#   sbatch --array=3 --exclude=gpu050 scripts/encoding_wordrate.sbatch   # probe
#   sbatch --exclude=gpu050 scripts/encoding_wordrate.sbatch
#
# At k>=256 the text band starts predicting primary auditory cortex
# (text|audio in Heschl 0% -> ~100% significant, GPT-2 and Llama alike).
# Either the LM predicts upcoming input, or long-context embeddings carry
# low-level temporal structure. Word rate is the cheapest such structure.
#
# Three bands, one design, one set of folds:
#
#   text       perlayer_llama8b_k1024:13
#   audio      perlayer_base_emotion:11
#   wordrate   wordrate
#
# and four models:
#
#   joint             all three
#   audio+wordrate    the kept model for --shuffle-block text
#   text+audio        so text|audio is measured in this same fit
#   audio
#
# Then:  r_joint - r_(audio+wordrate)   vs   r_(text+audio) - r_audio
# in Heschl. If the first collapses, the Heschl text signal is word rate.
#
# The four-band prediction fit cannot answer this: it has no audio+wordrate.

set -e

RUN_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "=== RUN ${RUN_ID} started $(date -Is) on $(hostname) ==="
echo "=== RUN ${RUN_ID} started $(date -Is) on $(hostname) ===" >&2
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

module load GCCcore/10.3.0 Python/3.9.5
. "$PROJECT/envs/py39/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export DATA_DIR=$PROJECT/data
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

TEXT=${TEXT:-perlayer_llama8b_k1024:13}
AUDIO=${AUDIO:-perlayer_base_emotion:11}
WORDRATE=${WORDRATE:-wordrate}
N_ITER=${N_ITER:-40}
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
TAG=${TAG:-llamawordrate}

for EV in $EVALS; do
    echo "=== $SUBJ | --eval $EV | n_iter=$N_ITER | $(date -Is) ==="
    # --n-splits is the OUTER fold count under cv; under holdout it IS the
    # alpha search, so it is passed only for cv.
    SPLITS=""
    [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        --band "text=${TEXT}" \
        --band "audio=${AUDIO}" \
        --band "${WORDRATE}=${WORDRATE}" \
        --models joint "audio+${WORDRATE}" text+audio audio \
        --tag "$TAG" \
        --stories-json "$STORIES_JSON" \
        --backend banded \
        --eval "$EV" \
        --min-ev 0.1 \
        --seed "$SEED" \
        --n-iter "$N_ITER" \
        --max-repeats 5 \
        $SPLITS
done

echo "=== RUN ${RUN_ID} $SUBJ done in $((SECONDS / 60)) min ==="
