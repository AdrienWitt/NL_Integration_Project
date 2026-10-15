#!/bin/bash
#SBATCH --job-name=emoctx_ext
#SBATCH --output=logs/emoctx_ext_%a_out
#SBATCH --error=logs/emoctx_ext_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:16G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --exclude=gpu050
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
# 2026-10-14. Emotion band with acoustic context (W s of preceding audio, only the
# readout frames pooled) x readout resolution (2 s per TR = paper grid; 0.5 s +
# Lanczos to TR centres, as the text band). Index -> (R, W).
set -e
cd ~/NL_Integration_Project
module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH=.
RS=(2 2 2 2 2 0.5 0.5 0.5 0.5 0.5)
WS=(0 2 4 8 16 0 2 4 8 16)
R=${R:-${RS[$SLURM_ARRAY_TASK_ID]}}; W=${W:-${WS[$SLURM_ARRAY_TASK_ID]}}  # R/W env: one extra point (--array=0)
echo "=== task $SLURM_ARRAY_TASK_ID : readout $R s, context $W s ==="
python -m extract.emotion_context --readout $R --context $W \
  --stories-json data/derivative/common_stories_all9.json
echo "=== done in $((SECONDS / 60)) min ==="
