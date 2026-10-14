#!/bin/bash
# 2026-10-13. Two analyses requested for the paper revision of Results:
#  (1) arousal + valence as a TWO-band model (dominance dropped: r = 0.963 with arousal),
#      fit -> Draper-Stoneman null per band (valence|arousal, arousal|valence), 1000 perms, nulls saved.
#  (2) Llama-3.1-8B every layer (0-32) at the context lengths that only had L13 (4, 16, 64, 512),
#      for the selection figure panel b; same recipe as the k256/k1024 per-layer sweeps
#      (fp32, common_stories_all9, gpt2_k256:8 reference row, gpu050 excluded).
set -e
cd ~/NL_Integration_Project
CC="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"

# ---- (1) arousal + valence -------------------------------------------------
AV="arousal=emotion_arousal valence=emotion_valence"
F=$(sbatch --parsable --exclude=gpu050 --output=logs/encbands_av_%a_out --error=logs/encbands_av_%a_err \
  --export=ALL,BANDS="$AV" scripts/cluster/encoding_bands.sbatch)
P=$(sbatch --parsable --exclude=gpu050 --dependency=aftercorr:$F \
  --output=logs/perm_av_%a_out --error=logs/perm_av_%a_err \
  --export=ALL,BANDS="$AV",SHUFFLE=each,FROM_DIR=results/encoding/bands_arousal-valence,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_null \
  scripts/cluster/permutation.sbatch)
echo "arousal+valence fit $F  nulls $P"

# ---- (2) Llama-8B all layers at k = 4 16 64 512 ---------------------------
M=NousResearch/Meta-Llama-3.1-8B
EXT=$(sbatch --parsable --exclude=gpu050 --time=06:00:00 --array=0-3 --gres=gpu:1,VramPerGpu:40G --constraint="$CC" \
  --output=logs/exsem_ll_%a_out --error=logs/exsem_ll_%a_err \
  --export=ALL,MODELS="$M $M $M $M",KS="4 16 64 512",LAYERS="0-32 0-32 0-32 0-32",PER_LAYER=1,STORIES=common_stories_all9.json,OUTNAMES="perlayer_llama8b_k4 perlayer_llama8b_k16 perlayer_llama8b_k64 perlayer_llama8b_k512",HF_HOME=$HOME/scratch/hf \
  scripts/cluster/extract_semantic.sbatch)
echo "extract $EXT"
for K in 4 16 64 512; do
  S=$(sbatch --parsable --exclude=gpu050 --time=12:00:00 --gres=gpu:1,VramPerGpu:40G --dependency=afterok:$EXT \
    --output=logs/semsw_ll${K}_%a_out --error=logs/semsw_ll${K}_%a_err \
    --export=ALL,LAYER_STORE=perlayer_llama8b_k$K,LAYER_LIST="$(seq -s ' ' 0 32)",EXTRA_SOURCES=perlayer_gpt2_k256:8,TAGSUF=_llama8b_k$K \
    scripts/cluster/semantic_sweep.sbatch)
  echo "k$K sweep $S"
done
echo SUBMITTED
