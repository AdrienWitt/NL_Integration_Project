#!/bin/bash
#SBATCH --job-name=enc_inter
#SBATCH --output=logs/encinter_%a_out
#SBATCH --error=logs/encinter_%a_err
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
# Does text x audio combine non-additively? The Clean_Irony interaction term.
# ===========================================================================
#
#   sbatch --array=3 scripts/encoding_interaction.sbatch      # probe, time it
#   sbatch scripts/encoding_interaction.sbatch
#
# Three bands, one design, one set of folds:
#
#   text    perlayer_gpt2_k256:8       the chosen semantic band
#   audio   perlayer_base_emotion:11   the chosen prosodic band
#   inter   inter_..._pc20             products of their top 20 PCs, 400 cols
#                                      (scripts/make_interaction_band.py)
#
# and two models, joint and text+audio, so that
#
#   r_joint - r_(text+audio) = what a NON-ADDITIVE combination explains
#                              beyond both bands
#
# Why: irony's combined model carried `semantic_* x prosody_*` product columns
# (--use_interaction) and ours was purely additive, so our `delta` measured
# co-location of two unique contributions rather than integration. This is the
# naturalistic analogue of that term. Its null is
#   permutation.sbatch with SHUFFLE=inter
# which shuffles the product band in time blocks while both parents stay
# aligned.
#
# N_ITER=40 for the same reason as encoding_prediction.sbatch: the random
# search is over the band-weight simplex, a 3-band joint against a 2-band
# subset, and a thin draw biases the difference DOWNWARD -- toward a false
# negative for the one claim this job tests. Keep it equal across subjects.

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

TEXT=${TEXT:-perlayer_gpt2_k256:8}
AUDIO=${AUDIO:-perlayer_base_emotion:11}
INTER=${INTER:-inter_perlayer_gpt2_k256L8_x_perlayer_base_emotionL11_pc20}
N_ITER=${N_ITER:-40}
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
TAG=${TAG:-interpc20}

for EV in $EVALS; do
    echo "=== $SUBJ | --eval $EV | inter=$INTER | n_iter=$N_ITER | $(date -Is) ==="
    # --n-splits is the OUTER fold count under cv; under holdout it IS the
    # alpha search, so it is passed only for cv.
    SPLITS=""
    [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        --band "text=${TEXT}" \
        --band "audio=${AUDIO}" \
        --band "inter=${INTER}" \
        --models joint text+audio \
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
