#!/bin/bash
#SBATCH --job-name=osmctx_sw
#SBATCH --output=logs/osmctx_sw_%a_out
#SBATCH --error=logs/osmctx_sw_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:24G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-8
#SBATCH --exclude=gpu050
# 2026-10-14. Scores every eGeMAPS variant as the sweep baseline band (one fit each),
# beside emotion L11, same settings as the paper selection sweep.
set -e
cd ~/NL_Integration_Project
module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
VARIANTS=${VARIANTS:-"opensmile_r2_w0 opensmile_r2_w2 opensmile_r2_w4 opensmile_r2_w8 opensmile_r2_w16 \
          opensmile_r0p5_w0 opensmile_r0p5_w2 opensmile_r0p5_w4 opensmile_r0p5_w8 opensmile_r0p5_w16"}
for V in $VARIANTS; do
  for EV in cv holdout; do
    python -m encoding.run_prosody_sweep --subjects "$SUBJ" --source perlayer_base_emotion \
      --configs "11" --stories-json common_stories_all9.json --baseline-features "$V" \
      --tag "${SUBJ}_$V" --eval "$EV" --n-splits 5 --min-ev 0.1 --himalaya-backend torch_cuda
  done
done
