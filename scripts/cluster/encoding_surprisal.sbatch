#!/bin/bash
#SBATCH --job-name=enc_surp
#SBATCH --output=logs/encsurp_%a_out
#SBATCH --error=logs/encsurp_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-8
#
# ===========================================================================
# The prediction band: surprisal and entropy against word rate, over k.
# ===========================================================================
#
#   sbatch scripts/cluster/encoding_surprisal.sbatch
#   KS="16 256" sbatch --array=3 scripts/cluster/encoding_surprisal.sbatch   # probe
#
# One task per subject; each loops the whole context grid, so a k is a run
# directory and not an array index. Requires scripts/split_surprisal_bands.py
# to have run -- the stores this fits do not come out of extract.surprisal
# directly, and the reason is the whole point of the job.
#
# Two bands, because word rate is not the nuisance it looks like
# ---------------------------------------------------------------
# Surprisal and entropy are word-level signals resampled onto TR times, so
# their amplitude tracks how many words arrived. On the design slice, 84
# stories, 28,088 TRs:
#
#     r(surprisal, word_rate)   k0 +0.976  k4 +0.841  k16 +0.766  k256 +0.740
#     r(entropy,   word_rate)   k0 +1.000  k4 +0.958  k16 +0.935  k256 +0.928
#
# Banded ridge gives one alpha per band, so a three-column store fit as one
# band shrinks the nuisance and the signal together and no contrast can pull
# them apart. Here `wordrate` carries its own alpha and
#
#     r_joint - r_wordrate = what surprisal and entropy add beyond word rate
#
# which is the same conditional-contribution logic the text/audio permutations
# use, one level down -- and the only form in which a surprisal claim survives
# the reviewer who says "that is just word rate".
#
# Why the k comparison needs the split too, not just the interpretation
# ---------------------------------------------------------------------
# word_rate is bit-identical across k (the splitter verifies it on all 84
# stories). Fit as one band, every k would be handed the same strong nuisance
# and k=0 would be handed the most of it -- r(entropy, word_rate) = 1.000
# there, because with no context the per-word entropy is a constant and a
# constant resampled by word arrival IS the word-rate signal. The argmax would
# land on the least contextual model for the least contextual reason. Split,
# `r_joint - r_wordrate` at each k is a comparison of the prediction signal
# alone and k=0 becomes what it was meant to be: the no-context floor.
#
# k is the experiment. Human expectation is built from a lossy memory of
# context, so the k that best predicts a region estimates that region's
# effective context depth; the argmax is the result, not a hyperparameter to
# settle first.
#
# What this job is NOT
# --------------------
# It is the selection sweep, not the reported model. The reported claim is
# what prediction adds beyond the semantic and prosodic bands already carried
# (perlayer_gpt2_k256:8, perlayer_base_emotion:11), which is a separate and
# far more expensive fit at the k chosen here -- and the permutation after it.
# Selecting that k on cv and reporting on holdout is this project's standing
# rule and the one it measured a cost for: cv and holdout argmaxes agreed in
# 1 of 36 prosody sweeps.
#
# The pe_gpt2_k<K> stores (768-d prediction-error vectors) are deliberately
# absent. They are a band of a different size and a different argument -- the
# port of ../Clean_Irony's incongruence vector -- and mixing them into the
# scalar grid would confound "which context length" with "scalar or vector".

set -e

# Run marker: --output/--error use fixed names by project convention, so this
# file holds the PREVIOUS run's text between submission and dispatch.
# SLURM_JOB_ID is the per-task id, not the id sbatch printed; anchor on this.
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

# Space-separated, never comma-delimited: `sbatch --export` splits its own
# value on commas, so an embedded one is silently eaten.
read -r -a KS_A <<< "${KS:-0 4 16 64 256}"
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
PRED_PREFIX=${PRED_PREFIX:-surp_pred_k}
WORDRATE=${WORDRATE:-wordrate}

for K in "${KS_A[@]}"; do
    for EV in $EVALS; do
        echo "=== $SUBJ | k=$K | --eval $EV | $(date -Is) ==="
        # --n-splits is the OUTER fold count under cv, where the alpha search
        # is the inner loop; under holdout it IS the alpha search, so passing
        # it there would select alphas on 5 folds for the reported number.
        SPLITS=""
        [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
        python3 -m encoding.run_encoding \
            --subjects "$SUBJ" \
            --band "${WORDRATE}=${WORDRATE}" \
            --band "pred=${PRED_PREFIX}${K}" \
            --models "$WORDRATE" pred joint \
            --tag "k${K}" \
            --stories-json "$STORIES_JSON" \
            --backend banded \
            --eval "$EV" \
            --min-ev 0.1 \
            --seed "$SEED" \
            --max-repeats 5 \
            $SPLITS
    done
done

echo "=== RUN ${RUN_ID} $SUBJ done in $((SECONDS / 60)) min ==="
