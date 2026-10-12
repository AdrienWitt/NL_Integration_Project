#!/bin/bash
#SBATCH --job-name=extract_surp
#SBATCH --output=logs/exsurp_%a_out
#SBATCH --error=logs/exsurp_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:16G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4
#
# ===========================================================================
# Prediction bands: surprisal, entropy and the prediction-error vector.
# ===========================================================================
#
#   sbatch scripts/extract_surprisal.sbatch
#
# One task per context length, k in {0, 4, 16, 64, 256}. Each writes two
# stores: surprisal_gpt2_k<K> (3 scalar columns) and pe_gpt2_k<K> (768-d).
#
# Why the whole grid rather than one k
# ------------------------------------
# Human expectations are built from a lossy memory of context, so the k that
# best predicts a region ESTIMATES that region's effective context depth. The
# argmax over k is the result, not a hyperparameter to settle first. k=0 comes
# along as the lexical-frequency nuisance regressor, which otherwise needs an
# external corpus: with no context the model's surprisal is close to unigram
# log-frequency, and measured on `souls` it correlates with the k=256 column at
# only r = 0.59 -- so contextual surprisal is demonstrably not word frequency.
#
# Note k=512 is deliberately absent. Surprisal saturates between 256 and 512
# the same way the encoding score does (16.40 -> 6.47 -> 5.59 bits at
# k=0/16/256 on `souls`), and GPT-2's 1024-token window caps usable context
# near 730 words anyway.
#
# There is no layer dimension here
# --------------------------------
# Surprisal and entropy come off the LM head, not a hidden state, so none of
# the k x layer grid that governs the semantic band applies. A k is an
# extraction, full stop. The pe_vector lives in the input embedding space,
# which is a space but not a choice.
#
# Runtime is not uniform, for extract_semantic.sbatch's reason
# ------------------------------------------------------------
# The block of words read out per pass is capped at stride_frac * k, so k=0, 4
# and 16 all come out at one forward pass per word -- about 160k tiny passes
# over 84 stories -- while k=256 reads 25 words per pass. The short-context
# tasks are the slow ones, which is the opposite of the intuition.
#
# --stride-frac MUST match what extract_semantic.sbatch used for the same k
# (0.1 everywhere so far), or "surprisal at k" and "representation at k" are
# not the same k and the two bands cannot be compared.
#
# Prerequisites: identical to extract_semantic.sbatch -- TextGrids on disk and
# GPT-2 in $HF_HOME, since jobs run with HF_HUB_OFFLINE=1.

set -e
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

LOG="${TMPDIR:-/tmp}/exsurp_${SLURM_ARRAY_TASK_ID}.log"
trap 'cp -f "$LOG" "logs/exsurp_${SLURM_ARRAY_TASK_ID}_full.log" 2>/dev/null || true' EXIT
exec > "$LOG" 2>&1

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export DATA_DIR=$PROJECT/data
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Space-separated and index-aligned with the array, never comma-delimited:
# `sbatch --export` splits its own value on commas, so an embedded one is
# silently eaten and a task writes to the wrong store. See the same note in
# extract_semantic.sbatch, where it cost a green job that produced no output.
read -r -a KS_A <<< "${KS:-0 4 16 64 256}"
read -r -a ST_A <<< "${STRIDES:-}"
read -r -a ON_A <<< "${OUTNAMES:-}"
# Another LM: MODEL=NousResearch/Meta-Llama-3.1-8B with HF_HOME=$HOME/scratch/hf
# and --gres VramPerGpu:40G. Same model for every task in the array.
MODEL=${MODEL:-gpt2}

K=${KS_A[$SLURM_ARRAY_TASK_ID]}
STRIDE=${ST_A[$SLURM_ARRAY_TASK_ID]:-0.1}
OUT=${ON_A[$SLURM_ARRAY_TASK_ID]:-surprisal_gpt2_k${K}}

# `[ x ] && y` is the wrong shape under `set -e`: a false test makes the line
# return 1 and kills the job before it starts.
PE_FLAG="--pe-vector"
if [ "${PE_VECTOR:-1}" = "0" ]; then
    PE_FLAG=""
fi

echo "=== RUN ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started $(date -Iseconds) on $(hostname) ==="
echo "=== task $SLURM_ARRAY_TASK_ID : $MODEL, context-words $K, stride-frac $STRIDE, "\
     "out $OUT ${PE_FLAG:-(scalars only)} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m extract.surprisal \
    --model "$MODEL" \
    --context-words "$K" \
    --stride-frac "$STRIDE" \
    $PE_FLAG \
    --out-name "$OUT"

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
