#!/bin/bash
#SBATCH --job-name=enc_pros
#SBATCH --output=logs/encpros_%a_out
#SBATCH --error=logs/encpros_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:32G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-8
#
# ===========================================================================
# The three prosody bands, side by side, through ONE runner.
# ===========================================================================
#
#   sbatch scripts/encoding_prosody_bands.sbatch
#
#   band                    dim    columns
#   emotion_avd               3         12    affect, named
#   opensmile                88        352    acoustics, hand-designed
#   perlayer_base_emotion:11 1024      4096    the learned representation
#
# Three levels of abstraction over one modality. The question the stage exists
# to answer is how much of what the audio band explains survives an affect-only
# bottleneck, and that is a ratio between the first and the third.
#
# Why not just read run_prosody_sweep's table
# -------------------------------------------
# The sweep already scored opensmile and every emotion layer on the corrected
# grid (2026-09-09) and those numbers stand -- but it calls
# `fit_banded_cv(inner_n_splits=args.n_splits)` while `run_encoding` leaves the
# inner CV unbounded, i.e. leave-one-story-out. Different alpha selection, so
# the two runners' cv scores are close but not the same quantity, and a table
# that mixes them is comparing procedures as well as bands. Everything here goes
# through run_encoding, so the rows differ only in the band.
#
# It also produces what the sweep does not: voxelwise maps. sweep.csv holds one
# row per configuration; the affective-fraction map (r_avd / r_emotion per
# voxel) needs the maps.
#
# One band per run, not one joint model
# -------------------------------------
# `--band NAME=STORE --models NAME` is a single-band banded ridge -- same solver,
# same alpha grid, same folds as every other row, which is the condition the
# banded.py docstring sets for the unimodal fits. Note this is deliberately NOT
# the `joint` model of scripts/encoding_avd.sbatch: there the three affective
# dimensions each get their own alpha, which is the right model for asking where
# each dimension lives, but a richer one than a single 3-column band. This job
# supplies the apples-to-apples row.

set -e

# ---------------------------------------------------------------------------
# Run marker. --output/--error use fixed names by project convention, so this
# file still holds the PREVIOUS run's text between submission and dispatch --
# and a reader in that window sees the past while believing it sees the present.
# That inverted a diagnosis twice in one session: once reading a replacement
# job's first line as the stuck job "recovering", once reading last night's
# `eigh failed` as this run's.
#
# Anchor on this line instead of on a timestamp:
#     sed -n "/=== RUN $SLURM_JOB_ID /,\$p" logs/<file>
echo "=== RUN ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} started $(date -Is) on $(hostname) ==="
echo "=== RUN ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} started $(date -Is) on $(hostname) ===" >&2
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

# Seeded since e39a6ba: an unseeded band-weight search made the same
# command irreproducible (max|diff| 2.9e-04 between two runs of one fit)
# and made runtime a lottery. Same seed for every band here, so the rows
# differ only in the band.
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}

# name=store, in the order they belong in the results table
BANDS=("avd=emotion_avd"
       "opensmile=opensmile"
       "emotion=perlayer_base_emotion:11")

for SPEC in "${BANDS[@]}"; do
    NAME=${SPEC%%=*}
    for EV in $EVALS; do
        echo "########## $SUBJ | $SPEC | --eval $EV : t+$((SECONDS / 60)) min"
        # --n-splits means two different things. Under --eval cv it sets the
        # OUTER folds, and the alpha search is the inner loop, left at
        # leave-one-story-out. Under --eval holdout there is no outer loop:
        # --n-splits IS the alpha search (fit_banded(..., splits=splits)), so
        # passing 5 there would select alphas on 5 folds for the number we
        # actually report, while the cv path searches over 23. Pass it only for
        # cv; a final single-configuration model can afford the extra folds,
        # which is what fit_banded_cv's docstring says they are for.
        SPLITS=""
        [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
        python3 -m encoding.run_encoding \
            --subjects "$SUBJ" \
            --band "$SPEC" \
            --models "$NAME" \
            --stories-json "$STORIES_JSON" \
            --backend banded \
            --eval "$EV" \
            --min-ev 0.1 \
        --seed "$SEED" \
            --max-repeats 5 \
            $SPLITS
    done
done

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
