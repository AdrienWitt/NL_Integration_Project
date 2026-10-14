#!/bin/bash
# Llama-3.1-8B L13 at k=64 and k=512: extraction -> joint (cv+holdout) -> conditional + marginalfixed nulls.
# Same settings as the k4/k16 grid (2026-10-05). Submitted 2026-10-13.
set -e
cd ~/NL_Integration_Project
CC="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
EXT=$(sbatch --parsable --exclude=gpu050 --array=0-1 --gres=gpu:1,VramPerGpu:40G --constraint="$CC" \
  --export=ALL,MODELS="NousResearch/Meta-Llama-3.1-8B NousResearch/Meta-Llama-3.1-8B",KS="64 512",LAYERS="13 13",STORIES=common_stories_all9.json,OUTNAMES="llama8b_k64L13 llama8b_k512L13",HF_HOME=$HOME/scratch/hf \
  scripts/cluster/extract_semantic.sbatch)
echo "extract $EXT"
for K in 64 512; do
  J=$(sbatch --parsable --exclude=gpu050 --dependency=afterok:$EXT --export=ALL,TEXT=llama8b_k${K}L13 scripts/cluster/encoding_joint.sbatch)
  C=$(sbatch --parsable --exclude=gpu050 --dependency=aftercorr:$J --export=ALL,TEXT=llama8b_k${K}L13,N_PERMS=1000,SHUFFLE=conditional,SAVE_NULL=1,TAGSUF=_null scripts/cluster/permutation.sbatch)
  M=$(sbatch --parsable --exclude=gpu050 --dependency=aftercorr:$J --export=ALL,TEXT=llama8b_k${K}L13,N_PERMS=1000,SHUFFLE=both,SAVE_NULL=1,TAGSUF=_marginalfixed scripts/cluster/permutation.sbatch)
  echo "k$K joint $J conditional $C marginalfixed $M"
done
echo SUBMITTED
