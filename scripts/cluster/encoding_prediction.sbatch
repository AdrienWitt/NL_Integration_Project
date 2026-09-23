#!/bin/bash
#SBATCH --job-name=enc_pred
#SBATCH --output=logs/encpred_%a_out
#SBATCH --error=logs/encpred_%a_err
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
# Does prediction add anything the representation bands do not already carry?
# ===========================================================================
#
#   K=256 sbatch --array=3 scripts/encoding_prediction.sbatch      # probe, time it
#   K=256 sbatch scripts/encoding_prediction.sbatch
#
# Four bands, one design, one set of folds:
#
#   text       perlayer_gpt2_k256:8       the chosen semantic band
#   audio      perlayer_base_emotion:11   the chosen prosodic band
#   pred       surp_pred_k<K>             surprisal + entropy at the k chosen
#                                         on cv by encoding_surprisal.sbatch
#   wordrate   wordrate                   the nuisance the other two ride on
#
# and three models:
#
#   joint                   all four
#   text+audio+wordrate     everything except prediction
#   text+audio              the published joint, refit here so the comparison
#                           shares folds and an alpha search with the rest
#
# so that
#
#   r_joint - r_(text+audio+wordrate) = what surprisal and entropy add beyond
#                                       the representations AND beyond word rate
#
# That is the whole claim. A prediction band that beats nothing once the
# semantic band is in the model is the expected outcome and a publishable one:
# GPT-2's hidden state at L8 already encodes what the LM head is about to
# predict, so the scalar may be strictly redundant with the 768 dimensions it
# was computed from. The test exists because "may be" is not a result.
#
# Why word_rate is a band and not a column
# ----------------------------------------
# Banded ridge assigns one alpha per band. Inside one band the nuisance and
# the signal are shrunk together and no contrast can separate them, and word
# rate is not a small nuisance here: r(surprisal, word_rate) = +0.74 at k=256
# and r(entropy, word_rate) = +0.93. Left in the pred band, a positive
# contribution would be unattributable.
#
# N_ITER is raised above the project default, on purpose
# -------------------------------------------------------
# `random_search` draws N_ITER points on the simplex of band WEIGHTS, so the
# same budget explores a 4-band simplex far more thinly than the 3-band subset
# it is being compared against. That asymmetry biases
# `r_joint - r_(text+audio+wordrate)` DOWNWARD -- the joint is the worse-tuned
# of the two nested models -- which is the direction that manufactures a false
# negative for the one claim this job exists to test. This file has already
# measured what a thin draw costs: two independent searches on identical data
# differed by 0.0002-0.0006 at two bands and +0.0036 at three.
#
# The cost is linear in N_ITER (fits = outer x inner x N_ITER) and is why the
# probe comes first. Keep the same N_ITER for every subject, or the group mean
# mixes two search budgets.
#
# Not comparable to the published joint by construction
# -----------------------------------------------------
# This joint has four bands where the published one has two, so its r is a
# different number and not an update to `results/encoding/perlayer_gpt2_k256L8
# __perlayer_base_emotionL11`. `text+audio` is refit here precisely so the
# within-run comparison exists without anyone having to cross runs.

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

: "${K:?set K to the context length chosen on cv, e.g. K=256}"
TEXT=${TEXT:-perlayer_gpt2_k256:8}
AUDIO=${AUDIO:-perlayer_base_emotion:11}
PRED=${PRED:-surp_pred_k${K}}
WORDRATE=${WORDRATE:-wordrate}
N_ITER=${N_ITER:-40}
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
TAG=${TAG:-predk${K}}

for EV in $EVALS; do
    echo "=== $SUBJ | --eval $EV | k=$K | n_iter=$N_ITER | $(date -Is) ==="
    # --n-splits is the OUTER fold count under cv; under holdout it IS the
    # alpha search, so it is passed only for cv.
    SPLITS=""
    [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        --band "text=${TEXT}" \
        --band "audio=${AUDIO}" \
        --band "pred=${PRED}" \
        --band "${WORDRATE}=${WORDRATE}" \
        --models joint "text+audio+${WORDRATE}" text+audio \
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
