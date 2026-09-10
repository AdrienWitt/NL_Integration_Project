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

# Everything that varies between arms is an env var, so a sweep does not need
# a copy of this file per arm:
#
#   sbatch --export=ALL,MODEL=emotion,EPOCHS=30,SAVE_LIMIT=40 scripts/finetune.sbatch
#   sbatch --export=ALL,MODEL=emotion,EXTRA="--l2sp 1e-2" scripts/finetune.sbatch
#
# SAVE_LIMIT is worth raising above the default 3. The damage fine-tuning does
# to brain prediction grows with how far the weights travel, so the drift/damage
# curve is readable *inside a single run* -- extract features from several
# epochs and score each. The two published runs cannot supply it: they kept
# save_total_limit=3, i.e. only epochs 29-32, all at the far end of the path.
MODEL=${MODEL:-wav2vec2-robust}
EPOCHS=${EPOCHS:-10}
PATIENCE=${PATIENCE:-3}
LR=${LR:-3e-5}
SAVE_LIMIT=${SAVE_LIMIT:-3}
SEED=${SEED:-42}

echo "=== arm: $MODEL, $EPOCHS epochs, save_total_limit=$SAVE_LIMIT ${EXTRA:+, extra: $EXTRA} ==="

torchrun --standalone --nproc_per_node="$NGPU" -m finetune.run_finetune \
    --model "$MODEL" \
    --batch-size 8 \
    --grad-accum "$ACCUM" \
    --num-epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --learning-rate "$LR" \
    --save-total-limit "$SAVE_LIMIT" \
    --metric-for-best eval_mean_r \
    --dataloader-workers "$WORKERS" \
    --seed "$SEED" \
    $EXTRA
