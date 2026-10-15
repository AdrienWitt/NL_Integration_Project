#!/bin/bash
#SBATCH --job-name=emoctx_sw
#SBATCH --output=logs/emoctx_sw_%a_out
#SBATCH --error=logs/emoctx_sw_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:24G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-89
#SBATCH --exclude=gpu050
# 2026-10-14. Layer sweep (blocks 0-11) of every emotion-context store, same
# settings as scripts/cluster/sweep_common_all_layers.sbatch (the paper selection sweep).
set -e
cd ~/NL_Integration_Project
module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
STORES=(perlayer_emoctx_r2_w0 perlayer_emoctx_r2_w2 perlayer_emoctx_r2_w4 perlayer_emoctx_r2_w8 perlayer_emoctx_r2_w16
        perlayer_emoctx_r0p5_w0 perlayer_emoctx_r0p5_w2 perlayer_emoctx_r0p5_w4 perlayer_emoctx_r0p5_w8 perlayer_emoctx_r0p5_w16)
[ -n "$STORE_LIST" ] && read -ra STORES <<< "$STORE_LIST"  # e.g. STORE_LIST=perlayer_emoctx_r0p5_w32 --array=0-8
NSUBJ=${#SUBJECTS[@]}
STORE=${STORES[$((SLURM_ARRAY_TASK_ID / NSUBJ))]}
SUBJ=${SUBJECTS[$((SLURM_ARRAY_TASK_ID % NSUBJ))]}
echo "=== task $SLURM_ARRAY_TASK_ID : $SUBJ / $STORE ==="
for EV in ${EVALS:-cv holdout}; do
  python -m encoding.run_prosody_sweep --subjects "$SUBJ" --source "$STORE" \
    --configs "0 1 2 3 4 5 6 7 8 9 10 11" --stories-json common_stories_all9.json \
    --baseline-features opensmile --tag "$SUBJ" --eval "$EV" --n-splits 5 \
    --min-ev 0.1 --himalaya-backend torch_cuda
done
echo "=== done in $((SECONDS / 60)) min ==="
