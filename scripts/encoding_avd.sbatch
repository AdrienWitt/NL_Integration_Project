#!/bin/bash
#SBATCH --job-name=enc_avd
#SBATCH --output=logs/encavd_%a_out
#SBATCH --error=logs/encavd_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
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
# Seven models, because a leave-one-out test needs all three subsets
# ------------------------------------------------------------------
#   joint   = A + D + V          the full model
#   A + V   -> r_joint - r_(A+V) = the unique contribution of D
#   D + V   -> r_joint - r_(D+V) = the unique contribution of A
#   A + D   -> r_joint - r_(A+D) = the unique contribution of V
#   A, D, V alone                 the standalone scores
#
# The first run fitted only A+V, which shows that dominance adds nothing but
# cannot show that the other two add something -- half the argument. Without
# the other two subsets, the test is indistinguishable from one that penalises
# any third band whatever it contains.
#
# This is the same conditional-contribution logic the text/audio permutations
# use, one level down. It is deliberately NOT a comparison against openSMILE or
# the emotion layer; those answer "which band predicts better", a different
# question with its own runs.
#
# What the first run already showed, and why the subsets matter: the unique
# contribution of dominance is +0.0227 on cv (9/9 subjects, sign test p=0.002)
# and -0.0107 on holdout (1/9, p=0.998). The sign flips. A band 96% redundant
# with one already present buys cv fit through the extra hyperparameter and
# loses it on a genuinely held-out story -- which is what "choose on cv, report
# on holdout" is for.
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
# --n-iter stays at the default 20. An earlier version of this file raised it to
# 200 on the reasoning that "each fit is seconds" at 12 columns. That was
# asserted, not measured, and all nine tasks hit the 4-hour limit: the cost of
# random_search is the number of fits, `outer * inner * n_iter`, and it scales
# with how many BANDS there are rather than how many columns, because what it
# samples is the simplex of band weights. Measured on UTS01 at n_iter=200:
#
#   arousal / dominance / valence   1 min 45 each  (one band: simplex is a point)
#   arousal+valence                 > 3 h 54       (two bands, never finished)
#
# 5 outer x 23 inner x 200 = 23,000 fits per model, against 500 for a sweep.
# Two factors had been multiplied in the same change -- n_iter (x10) and an
# unbounded inner loop (x4.6) -- each argued for on its own merits.
#
# The inner loop stays leave-one-story-out, matching every other final model
# here; scripts/inner_cv_check.sbatch measured 5-fold against it at |diff| =
# 0.0001, so bounding it would be defensible too, but consistency with
# encoding_prosody_bands.sbatch is worth more than the 4.6x.
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

# --seed reaches the band-weight search. It was unseeded until e39a6ba, which
# is why nine identical designs ran between 1h13 and 5h27: random_search draws
# n_iter points on the simplex, the kernel factorised is sum_i exp(delta_i)K_i,
# and the draw decides whether eigh converges or the 42x svd fallback fires.
#
# The flip side: with one seed and one design, every subject now draws the SAME
# points, so either none of them hits the fallback or all nine do. Watch the
# first minutes for "eigh failed" (it logs since 7eb9882); if it fires, cancel,
# change SEED, resubmit. --primal sidesteps the question entirely and is ~53x
# faster here, at the cost of differing from the solver every published number
# used and of being unusable by run_permutation --from-encoding.
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}

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
        --models arousal dominance valence \
                 dominance+valence arousal+dominance arousal+valence \
                 joint \
        --stories-json "$STORIES_JSON" \
        --backend banded \
        --eval "$EV" \
        --min-ev 0.1 \
        --seed "$SEED" \
        --max-repeats 5 \
        $SPLITS
done

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
