#!/bin/sh
#SBATCH --job-name=ft_prosody
#SBATCH --output=ft_prosody_out
#SBATCH --error=ft_prosody_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4,VramPerGpu:20G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -e

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
WORKERS=$(( SLURM_CPUS_PER_TASK / NGPU ))
ACCUM=$(( 4 / NGPU ))
[ "$WORKERS" -lt 1 ] && WORKERS=1
[ "$ACCUM" -lt 1 ] && ACCUM=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

torchrun --standalone --nproc_per_node="$NGPU" -m finetune.run_finetune \
    --model wav2vec2-robust \
    --batch-size 8 \
    --grad-accum "$ACCUM" \
    --num-epochs 10 \
    --patience 3 \
    --learning-rate 3e-5 \
    --metric-for-best eval_mean_r \
    --dataloader-workers "$WORKERS" \
    --seed 42
