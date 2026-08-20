#!/bin/sh
#SBATCH --job-name=ft_prosody
#SBATCH --output=logs/ft_%j.out
#SBATCH --error=logs/ft_%j.err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#
# --- GPUs -----------------------------------------------------------------
# One task; torchrun spawns one process per GPU underneath it. Do NOT use
# --ntasks=N here: that would start N independent copies of the script, each
# of which would then launch its own torchrun.
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# bf16 needs Ampere or newer. shared-gpu mixes Pascal (6.x), Turing (7.5) and
# Ampere+ nodes, so constrain it — otherwise the run silently falls back to
# fp16 on older cards. public-gpu is Pascal only (titan_x / P100), so it is
# NOT a drop-in alternative despite its longer 2-day limit.
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"

set -e

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"
mkdir -p logs

module load GCCcore/12.3.0 Python/3.11.3
source "$PROJECT/envs/prosody/bin/activate"

export PYTHONUNBUFFERED=1
export DATA_DIR=$PROJECT/data
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1          # checkpoints must be pre-fetched on the login node
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
WORKERS=$(( SLURM_CPUS_PER_TASK / NGPU ))
[ "$WORKERS" -lt 1 ] && WORKERS=1

echo "node        : $(hostname)"
echo "gpus        : $NGPU ($CUDA_VISIBLE_DEVICES)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus        : $SLURM_CPUS_PER_TASK  -> $WORKERS dataloader workers/rank"

# Effective batch = per-device x n_gpus x grad-accum. The single-GPU default is
# 8 x 4 = 32; dividing grad-accum by the GPU count holds that constant so
# adding GPUs buys speed rather than silently changing the optimisation.
ACCUM=$(( 4 / NGPU ))
[ "$ACCUM" -lt 1 ] && ACCUM=1
echo "effective batch: $(( 8 * NGPU * ACCUM ))"

torchrun --standalone --nproc_per_node="$NGPU" -m finetune.run_finetune \
    --model wav2vec2-robust \
    --freeze-layers auto \
    --batch-size 8 \
    --grad-accum "$ACCUM" \
    --num-epochs 10 \
    --patience 3 \
    --metric-for-best eval_mean_r \
    --dataloader-workers "$WORKERS" \
    --seed 42
