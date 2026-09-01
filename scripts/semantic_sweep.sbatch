#!/bin/bash
#SBATCH --job-name=sem_sweep
#SBATCH --output=logs/semsw_%a_out
#SBATCH --error=logs/semsw_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:24G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-8
#
# ===========================================================================
# Semantic context-length sweep: nine subjects, one GPU each, both evals.
# ===========================================================================
#
#   sbatch --dependency=afterok:<extract_jobid> scripts/semantic_sweep.sbatch
#
# One task per subject, because every configuration for a subject shares that
# subject's responses, EV mask and CV folds -- loading those is the slowest
# step in the pipeline, so splitting a subject across tasks would pay for it
# repeatedly. Nine tasks is also the natural width: the sweep is over context
# lengths, and those are cheap once the responses are in memory.
#
# Common stories only
# -------------------
# common_stories_all9.json puts all nine subjects on the same 24 training
# stories, so scores can be averaged across subjects. Per-subject lists
# (26-84 stories) would make a group mean part semantics and part "UTS01
# heard three times more data", and are what made the all-stories run OOM.
#
# Both evaluations
# ----------------
# --eval cv       cross-validated inside the training stories. The honest
#                 basis for CHOOSING a context length.
# --eval holdout  scored on wheretheressmoke. The number to REPORT for a
#                 choice already made. Its standard errors are ~4x the
#                 cross-validated ones on a 291-TR story, so read it as a
#                 shape and take the selection from the cv run.
#
# The baseline row is the existing gpt2_mean band, scored under identical
# folds and mask. That is the zero line: it says whether contextualising at
# all beats the isolated-word band every result so far has used.

set -e
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

# Log to node-local disk, copy once at exit -- BeeGFS returns Errno 121 on the
# steady trickle of writes a long job makes, and under `set -e` a failed echo
# is fatal. See sweep_common_all_layers.sbatch for the full story.
LOG="${TMPDIR:-/tmp}/semsw_${SLURM_ARRAY_TASK_ID}.log"
trap 'cp -f "$LOG" "logs/semsw_${SLURM_ARRAY_TASK_ID}_full.log" 2>/dev/null || true' EXIT
exec > "$LOG" 2>&1

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

SOURCES="gpt2_k0 gpt2_k1 gpt2_k4 gpt2_k16 gpt2_k64 gpt2_k256"

RESP="data/ds003020/derivative/preprocessed_data/$SUBJ"
if [ ! -d "$RESP" ]; then
    echo "ERROR: no responses for $SUBJ at $RESP"
    exit 1
fi
for S in $SOURCES; do
    if [ ! -d "data/features/$S" ]; then
        echo "ERROR: missing feature store data/features/$S"
        echo "       run scripts/extract_semantic.sbatch first"
        exit 1
    fi
done

echo "=== task $SLURM_ARRAY_TASK_ID : $SUBJ ==="
echo "    sources: $SOURCES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_eval () {   # eval mode
    echo "########## $SUBJ / --eval $1 : t+$((SECONDS / 60)) min"
    python -m encoding.run_semantic_sweep \
        --subjects "$SUBJ" \
        --sources "$SOURCES" \
        --stories-json common_stories_all9.json \
        --baseline-features gpt2_mean \
        --tag "$SUBJ" \
        --eval "$1" \
        --n-splits 5 \
        --min-ev 0.1 \
        --himalaya-backend torch_cuda
}

run_eval cv
run_eval holdout

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
