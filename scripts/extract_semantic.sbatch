#!/bin/bash
#SBATCH --job-name=extract_sem
#SBATCH --output=logs/exsem_%a_out
#SBATCH --error=logs/exsem_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:16G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5
#
# ===========================================================================
# The semantic context-length bands: one store per k, all 84 stories.
# ===========================================================================
#
#   sbatch scripts/extract_semantic.sbatch
#
# One task per context length. k=0 embeds each word alone, which reproduces
# the isolated-word baseline the current `gpt2_mean` band is; every larger k
# gives each word that many preceding words inside one causal forward pass.
#
# Why k=0 is the slowest task
# ---------------------------
# `extract.context_lm` caps the block of words read out per forward pass at
# `stride_frac * k`, because in a causal LM every word in a block also attends
# to the earlier words of that block. Without the cap, k=0 and k=256 would
# both give each word hundreds of words of context and the sweep would return
# a flat line. The cost is that k=0 means one forward pass per word — about
# 140k tiny passes over 84 stories. Still minutes on a GPU, but it is why
# this array is not uniform in runtime.
#
# Prerequisites on the cluster (login node, it has internet):
#   - data/ds003020/derivative/TextGrids/   (84 files, ~850 KB)
#   - GPT-2 in $HF_HOME: jobs run with HF_HUB_OFFLINE=1, so from_pretrained
#     must never touch the network.

set -e
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

LOG="${TMPDIR:-/tmp}/exsem_${SLURM_ARRAY_TASK_ID}.log"
trap 'cp -f "$LOG" "logs/exsem_${SLURM_ARRAY_TASK_ID}_full.log" 2>/dev/null || true' EXIT
exec > "$LOG" 2>&1

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export DATA_DIR=$PROJECT/data
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1

KS=(0 1 4 16 64 256)
K=${KS[$SLURM_ARRAY_TASK_ID]}

echo "=== task $SLURM_ARRAY_TASK_ID : context-words $K ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m extract.context_lm \
    --model gpt2 \
    --context-words "$K" \
    --layers last \
    --out-name "gpt2_k${K}"

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
