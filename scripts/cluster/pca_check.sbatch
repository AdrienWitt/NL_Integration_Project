#!/bin/bash
#SBATCH --job-name=pca_check
#SBATCH --output=logs/pcachk_%a_out
#SBATCH --error=logs/pcachk_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:24G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --exclude=gpu050
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-8
#
# ===========================================================================
# Does PCA on the semantic band cost encoding performance?
# ===========================================================================
#
#   sbatch --export=ALL,TEXT=perlayer_llama8b_k1024:13 scripts/cluster/pca_check.sbatch
#
# One single-band text model per condition, everything else identical: same
# subject, folds, EV mask and seed, so the rows differ only in the projection.
# A single band means the band-weight simplex is a point and the alpha search
# is cheap (~2 min a model), which is why this is affordable at all.
#
# Read holdout as the clean test. Under --eval cv the PCA is fitted once on all
# 24 training stories before the outer folds (run_encoding builds the design
# first), so each fold's test stories took part in fitting the projection. That
# is unsupervised -- Y is never seen -- but it is transductive, and it can only
# flatter PCA. Under --eval holdout the projection is fitted on the training
# stories and applied unchanged to wheretheressmoke.
#
# Compare with:
#   PYTHONPATH=. python scripts/summarise_pca_check.py

set -e
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

LOG="${TMPDIR:-/tmp}/pcachk_${SLURM_ARRAY_TASK_ID}.log"
trap 'cp -f "$LOG" "logs/pcachk_${SLURM_ARRAY_TASK_ID}_full.log" 2>/dev/null || true' EXIT
exec > "$LOG" 2>&1

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
TEXT=${TEXT:-perlayer_llama8b_k1024:13}
# "none" is the unprojected band. Integers are components, a fraction <= 1 is
# explained variance. 1024 matches the audio band's width.
CONDS=${CONDS:-"none 1024 256 0.9"}

echo "=== RUN ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started $(date -Iseconds) on $(hostname) ==="
echo "=== $SUBJ | text=$TEXT | conditions: $CONDS ==="

for C in $CONDS; do
    PCA_ARGS=""
    TAG="pca_none"
    if [ "$C" != "none" ]; then
        PCA_ARGS="--use-pca --n-comps $C"
        TAG="pca_${C}"
    fi
    for EV in cv holdout; do
        echo "########## $SUBJ | $TAG | --eval $EV : t+$((SECONDS / 60)) min"
        SPLITS=""
        [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
        python3 -m encoding.run_encoding \
            --subjects "$SUBJ" \
            --band "text=$TEXT" \
            --models text \
            --stories-json common_stories_all9.json \
            --backend banded \
            --eval "$EV" \
            --min-ev 0.1 \
            --max-repeats 5 \
            --seed 0 \
            --tag "$TAG" \
            $PCA_ARGS $SPLITS
    done
done

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
