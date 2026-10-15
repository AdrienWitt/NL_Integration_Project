#!/bin/bash
# 2026-10-15. Integration as a function of the AUDIO context, text fixed at the selected
# Llama-8B L13 k512: emotion band (L11, 0.5-s readout) with W = 0, 2, 4, 8 s of preceding
# audio. W = 16 is the main fit (submit_acoustic_ctx.sh), W = 32 follows if extracted.
# Joint (cv + holdout) + conditional nulls only: delta and its two conditional terms.
# Together with the text-context series at W = 16 this is a cross through the selected point.
set -e
cd ~/NL_Integration_Project
T=llama8b_k512L13
for W in ${WS:-0 2 4 8}; do
  A=perlayer_emoctx_r0p5_w${W}:11
  J=$(sbatch --parsable --exclude=gpu050 --output=logs/encjoint_aw${W}_%a_out --error=logs/encjoint_aw${W}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$A scripts/cluster/encoding_joint.sbatch)
  C=$(sbatch --parsable --exclude=gpu050 --dependency=aftercorr:$J --output=logs/perm_aw${W}_%a_out --error=logs/perm_aw${W}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$A,N_PERMS=1000,SHUFFLE=conditional,SAVE_NULL=1,TAGSUF=_null scripts/cluster/permutation.sbatch)
  echo "W=$W: joint $J conditional $C"
done
