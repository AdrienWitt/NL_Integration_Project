#!/bin/sh
#SBATCH --job-name=extract
#SBATCH --output=logs/ex_%x_%j.out
#SBATCH --error=logs/ex_%x_%j.err
#SBATCH --partition=shared-gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Inference only — no bf16, so no compute-capability constraint needed here.
#
#   sbatch --export=ALL,ARM=base    scripts/extract.sbatch   # frozen control
#   sbatch --export=ALL,ARM=robust  scripts/extract.sbatch   # after fine-tuning
#   sbatch --export=ALL,ARM=emotion scripts/extract.sbatch

set -e
ARM=${ARM:-base}
SEED=${SEED:-42}
LR=${LR:-3e-5}

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"
mkdir -p logs

module load GCCcore/12.3.0 Python/3.11.3
. "$PROJECT/envs/prosody/bin/activate"

export PYTHONUNBUFFERED=1
export DATA_DIR=$PROJECT/data
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1

FT=results/finetune

case "$ARM" in
  # THE CONTROL: no fine-tuning at all. Costs no training and is the
  # reference every fine-tuned arm has to beat. Registry default reads
  # mid-network (12-17), where prosody peaks in a *base* model.
  base)
      ARGS="--model wav2vec2-robust --layers auto --out-name base_robust" ;;
  base_emotion)
      ARGS="--model emotion --layers auto --out-name base_emotion" ;;

  # Fine-tuned checkpoints read HIGH: fine-tuning on prosody targets retunes
  # the upper layers, so the base default no longer applies.
  robust)
      ARGS="--model $FT/wav2vec2_robust_frozen_12_lr${LR}_seed${SEED}/final_model \
            --layers 18-23 --out-name ft_robust_18to23" ;;
  emotion)
      ARGS="--model $FT/emotion_frozen_6_lr${LR}_seed${SEED}/final_model \
            --layers 6-11 --out-name ft_emotion_6to11" ;;
  trunc12)
      ARGS="--model $FT/wav2vec2_robust_trunc12_frozen_6_lr${LR}_seed${SEED}/final_model \
            --layers 6-11 --out-name ft_trunc12_6to11" ;;

  # Prosody baseline band for the encoding stage (CPU-fine, but harmless here).
  opensmile)
      ARGS="" ; python3 -m extract.opensmile --out-name opensmile ; exit 0 ;;

  *) echo "unknown ARM: $ARM" >&2; exit 1 ;;
esac

echo "arm: $ARM"; nvidia-smi --query-gpu=name --format=csv,noheader
python3 -m extract.wav2vec $ARGS
