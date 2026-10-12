#!/bin/bash
#SBATCH --job-name=enc_bands
#SBATCH --output=logs/encbands_%a_out
#SBATCH --error=logs/encbands_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-8
#
# Any set of named bands, fit together through one runner (`--band NAME=STORE`).
# Written for the audio-band comparisons (2026-10-09):
#
#   sbatch --output=logs/encbands_emo_%a_out --error=logs/encbands_emo_%a_err \
#          --export=ALL,BANDS="emotion=perlayer_base_emotion:11 opensmile=opensmile" \
#          scripts/cluster/encoding_bands.sbatch
#   sbatch --output=logs/encbands_avd_%a_out --error=logs/encbands_avd_%a_err \
#          --export=ALL,BANDS="opensmile=opensmile avd=emotion_avd" \
#          scripts/cluster/encoding_bands.sbatch
#
# then `permutation.sbatch` with the same BANDS, SHUFFLE=each and
# FROM_DIR=results/encoding/bands_<names> gives one Draper-Stoneman null per
# band: "this band adds nothing beyond the others".
#
# Pass --output/--error on the command line when two arrays run at once, or
# their logs overwrite each other. MODELS is optional; with two bands the
# default (each band alone + joint) is everything `each` needs.

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

SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
BANDS=${BANDS:?set BANDS=\"name=store name=store ...\"}
MODELS=${MODELS:-}

BAND_ARGS=()
for b in $BANDS; do BAND_ARGS+=(--band "$b"); done
if [ -n "$MODELS" ]; then BAND_ARGS+=(--models $MODELS); fi

echo "=== $SUBJ | bands: $BANDS | models: ${MODELS:-default} ==="

for EV in $EVALS; do
    echo "########## $SUBJ | --eval $EV : t+$((SECONDS / 60)) min"
    # --n-splits only for cv: under holdout it IS the alpha search (see
    # encoding_joint.sbatch).
    SPLITS=""
    [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        "${BAND_ARGS[@]}" \
        --stories-json "$STORIES_JSON" \
        --backend banded \
        --eval "$EV" \
        --min-ev 0.1 \
        --max-repeats 5 \
        --seed "$SEED" \
        $SPLITS
done

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
