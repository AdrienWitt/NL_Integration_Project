#!/bin/bash
#SBATCH --job-name=enc_joint
#SBATCH --output=logs/encjoint_%a_out
#SBATCH --error=logs/encjoint_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-8
#
# ===========================================================================
# THE final model: the chosen semantic band against the chosen prosodic band.
# ===========================================================================
#
#   sbatch scripts/encoding_joint.sbatch
#
#   text   perlayer_gpt2_k16:8        768d   3,072 columns
#   audio  perlayer_base_emotion:11  1024d   4,096 columns
#   joint  both                              7,168 columns
#
# Every other encoding job in this repo was a *selection* run: sweeps that
# ranked layers, context lengths and bands so that this one could be fit once
# with the choices already made. Both bands were chosen on cross-validation
# inside the training stories, and neither selection ever touched
# `wheretheressmoke`, so `--eval holdout` here is genuinely out of sample.
#
# What it produces, per subject, that nothing else does
# -----------------------------------------------------
#   text_corrs / audio_corrs / joint_corrs   the three maps
#   joint_split_corrs                        each band's share inside the joint
#   joint_deltas                             the band weights
#
# and from those, the two headline statistics of the paper:
#
#   delta      = r_joint - max(r_text, r_audio)    integration
#   preference = r_text - r_audio                  which modality a voxel wants
#
# `joint_deltas` is not a diagnostic. `stats/run_permutation.py
# --from-encoding` reuses it so that the observed statistic and the null come
# out of the same estimator rather than out of two alpha searches that agree;
# that invariant already failed once (d713d63). Which is also why this job must
# NOT be run with `--primal`: primal deltas weight feature groups, dual ones
# weight kernels, and `load_observed_fit` refuses the mix rather than silently
# building a null from a model that was never fitted.
#
# Band names are `text` and `audio`, not `--band NAME=STORE`
# ----------------------------------------------------------
# `--band` exists for within-modality decompositions (the AVD job). Here the
# two-band default is what is wanted, and it matters beyond taste:
# `MODEL_BANDS` is imported by `stats.run_permutation`, which is text/audio by
# construction and looks for exactly those names in `meta.json`. Passing
# `--text-features` / `--audio-features` is what makes the permutation able to
# find this fit.
#
# One output root, nine subject subdirectories
# --------------------------------------------
# No `--tag $SUBJ`: `run_encoding` already nests by subject, and
# `--from-encoding` wants the run root with all nine under it. The array tasks
# write disjoint subdirectories of one tree.
#
#   results/encoding/perlayer_gpt2_k16L8__perlayer_base_emotionL11/
#       banded/{cv,holdout}/UTS0*/
#
# (The colon of `store:layer` becomes an `L` in the directory name only; the
# band stores go into meta.json unchanged, which is what a reader should key
# on.)
#
# On the 12 hour limit
# --------------------
# The single-band `enc_pros` tasks took 8-14 min for cv+holdout at p=4,096.
# This is three models over two evals at up to p=7,168, and p=7,168 against
# n=8,683 training TRs is deeper into the rank-deficient regime that makes
# `eigh` fail — the automatic `svd` fallback is ~42x slower, and one task
# hitting it is the difference between twenty minutes and half a day. The
# fallback now announces itself in the log (7eb9882), so check for
# "eigh failed" before concluding a task is hung. `--seed` is what decides
# whether it fires, so a task that trips it can be re-run at a different SEED
# rather than waited out.

set -e

# Run marker. --output/--error use fixed names by project convention, so this
# file still holds the PREVIOUS run's text between submission and dispatch.
# Anchor on this line rather than on a timestamp:
#     sed -n "/=== RUN <the id sbatch printed>_/,\$p" logs/<file>
# SLURM_JOB_ID is the per-TASK id inside an array, not the id sbatch printed,
# so anchoring on the submitted id never matched. SLURM_ARRAY_JOB_ID is that id;
# fall back to SLURM_JOB_ID for a non-array run, where they are the same thing.
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

# Same seed as every other encoding job, so this fit differs from the
# single-band rows only in the design and not in the draw.
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
TEXT=${TEXT:-perlayer_gpt2_k16:8}
AUDIO=${AUDIO:-perlayer_base_emotion:11}

echo "=== $SUBJ | text=$TEXT | audio=$AUDIO | seed=$SEED ==="

for EV in $EVALS; do
    echo "########## $SUBJ | --eval $EV : t+$((SECONDS / 60)) min"
    # --n-splits means two different things. Under --eval cv it sets the OUTER
    # folds and the alpha search stays at leave-one-story-out. Under --eval
    # holdout there is no outer loop: --n-splits IS the alpha search, so
    # passing it there would select alphas on 5 folds for the number actually
    # reported, while the cv path searches over 23. Pass it only for cv.
    SPLITS=""
    [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        --text-features "$TEXT" \
        --audio-features "$AUDIO" \
        --stories-json "$STORIES_JSON" \
        --backend banded \
        --eval "$EV" \
        --min-ev 0.1 \
        --max-repeats 5 \
        --seed "$SEED" \
        $SPLITS
done

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
