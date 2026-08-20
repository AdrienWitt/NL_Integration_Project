#!/bin/sh
#SBATCH --job-name=ft_prosody
#SBATCH --output=logs/ft_%x_%j.out
#SBATCH --error=logs/ft_%x_%j.err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#
# --- resources ------------------------------------------------------------
# One task; torchrun spawns one process per GPU underneath. Do NOT raise
# --ntasks: that starts N copies of this script, each launching its own
# torchrun.
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4,VramPerGpu:20G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
# VramPerGpu:20G excludes the 10 GB RTX 3080 nodes. The constraint pins
# Ampere or newer, which is what bf16 needs — shared-gpu also carries 6.x and
# 7.5 nodes, where the run would silently fall back to fp16. public-gpu is
# Pascal only, so its 2-day limit is no help to us.
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"

set -e

# Which arm to run:  sbatch --export=ALL,ARM=emotion scripts/finetune.sbatch
ARM=${ARM:-robust}

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"
mkdir -p logs

module load GCCcore/12.3.0 Python/3.11.3
. "$PROJECT/envs/prosody/bin/activate"

export PYTHONUNBUFFERED=1
export DATA_DIR=$PROJECT/data
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1          # checkpoints pre-fetched by setup_cluster.sh
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1         # torchrun ranks must not each grab all cores

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
WORKERS=$(( SLURM_CPUS_PER_TASK / NGPU ))
[ "$WORKERS" -lt 1 ] && WORKERS=1

# Effective batch = per-device x n_gpus x grad-accum. The single-GPU default
# is 8 x 4 = 32; scaling grad-accum down holds that constant, so extra GPUs
# buy wall-clock rather than silently changing the optimisation.
ACCUM=$(( 4 / NGPU ))
[ "$ACCUM" -lt 1 ] && ACCUM=1

echo "arm         : $ARM"
echo "node        : $(hostname)"
echo "gpus        : $NGPU ($CUDA_VISIBLE_DEVICES)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus        : $SLURM_CPUS_PER_TASK -> $WORKERS workers/rank"
echo "eff. batch  : $(( 8 * NGPU * ACCUM ))"
echo

case "$ARM" in
  # ARM A — the main 24-layer arm. freeze auto = bottom 12 of 24,
  # so 12 transformer layers train (~150M params).
  robust)   MODEL_ARGS="--model wav2vec2-robust" ;;

  # ARM B — the model of interest. Pruned to 12 layers, freeze auto = 6.
  emotion)  MODEL_ARGS="--model emotion" ;;

  # ARM C — depth-matched control for B: same base as the emotion model,
  # cut to 12 layers, so emotion pretraining is the only difference left.
  trunc12)  MODEL_ARGS="--model wav2vec2-robust --truncate-layers 12" ;;

  # Softer alternative to a hard freeze boundary: every layer stays plastic,
  # lower ones just train ~10x slower.
  llrd)     MODEL_ARGS="--model wav2vec2-robust --freeze-layers none --llrd 0.9" ;;

  *) echo "unknown ARM: $ARM (robust|emotion|trunc12|llrd)" >&2; exit 1 ;;
esac

torchrun --standalone --nproc_per_node="$NGPU" -m finetune.run_finetune \
    $MODEL_ARGS \
    --batch-size 8 \
    --grad-accum "$ACCUM" \
    --num-epochs 10 \
    --patience 3 \
    --learning-rate 3e-5 \
    --metric-for-best eval_mean_r \
    --dataloader-workers "$WORKERS" \
    --seed 42
