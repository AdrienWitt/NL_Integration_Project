#!/bin/bash
#SBATCH --job-name=enc_avd
#SBATCH --output=logs/encavd_%a_out
#SBATCH --error=logs/encavd_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-8
#
# ===========================================================================
# Arousal / dominance / valence as three bands of one banded ridge.
# ===========================================================================
#
#   sbatch --dependency=afterok:<avd jobid> scripts/encoding_avd.sbatch
#   EVALS="cv holdout" sbatch scripts/encoding_avd.sbatch
#
# One task per subject. Each fits five models from ONE design matrix and ONE
# set of folds, which is what makes the contrasts about dimensions rather than
# fold assignment:
#
#   arousal, dominance, valence     each dimension alone   (4 columns each)
#   arousal+valence                 the two-dimension model
#   joint                           all three
#
# The pair that answers "two dimensions or three"
# -----------------------------------------------
# r_joint - r_(arousal+valence) is the unique contribution of dominance beyond
# the other two -- the same conditional-contribution logic the text/audio
# permutations use, one level down. It is deliberately NOT a comparison against
# openSMILE or the emotion layer; those answer "which band predicts better",
# which is a different question with its own runs.
#
# Twelve columns is not a problem
# -------------------------------
# 3 dimensions x 4 FIR delays = 12 columns against ~8,700 training TRs. That is
# the easy regime; the hard one is p=4096, which is rank-deficient against
# n=9,461 and is what forces the svd fallback. Each band still gets its own
# alpha, so a low-dimensional band is not penalised for being small -- it gets
# a small alpha, the 4096-column bands get large ones, and every model is
# regularised at its own optimum.
#
# --n-iter is raised well above the sweep's 20. random_search samples the
# simplex of band weights, and 20 samples were calibrated for p=4096 where each
# fit is expensive. Here each fit is seconds, and the band weights are exactly
# what two collinear dimensions are fighting over, so sampling them well is the
# whole point.
#
# Common stories, because these maps get compared across subjects. Per-subject
# story lists would make a group difference part affect and part "UTS01 heard
# three times more data".

set -e
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

EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
NITER=${NITER:-200}

for EV in $EVALS; do
    echo "=== $SUBJ | --eval $EV ==="
    # --n-splits means two different things. Under --eval cv it sets the OUTER
    # folds, and the alpha search is the inner loop, left at leave-one-story-out.
    # Under --eval holdout there is no outer loop: --n-splits IS the alpha
    # search (fit_banded(..., splits=splits)), so passing 5 there would select
    # alphas on 5 folds for the number we actually report, while the cv path
    # searches over 23. Pass it only for cv; a final single-configuration model
    # can afford the extra folds, which is exactly what fit_banded_cv's
    # docstring says they are for.
    SPLITS=""
    [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        --band arousal=emotion_arousal \
        --band dominance=emotion_dominance \
        --band valence=emotion_valence \
        --models arousal dominance valence arousal+valence joint \
        --stories-json "$STORIES_JSON" \
        --backend banded \
        --eval "$EV" \
        --min-ev 0.1 \
        --max-repeats 5 \
        $SPLITS \
        --n-iter "$NITER"
done

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
