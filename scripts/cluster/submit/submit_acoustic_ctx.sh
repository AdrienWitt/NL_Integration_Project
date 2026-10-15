#!/bin/bash
# 2026-10-15. Acoustic bands move to the configurations chosen by the
# readout x context sweeps (emoctx/osmctx, selection rule cv mean + >= 8/9):
#   emotion  perlayer_emoctx_r0p5_w16:11  (0.5-s readout, 16 s of preceding audio)
#   AVD      perlayer_emoctx_r0p5_w16_avd (head of the same network)
#   eGeMAPS  opensmile_r0p5_w2            (functionals over [t-2, t+0.5], every 0.5 s)
# Everything the paper fits with an audio band is redone on these stores, with
# the same settings as the published runs. Joint fits get new directory names
# from their stores; the named-band fits get TAG=ctx, their nulls _ctx* tags.
# Selection rule = best cv mean (2026-10-15): emotion W=16 (.16972) over W=8
# (.16969); the first submission at W=8 was cancelled. SKIP_OSM=1 skips the
# text x eGeMAPS chain, which does not depend on the emotion band.
set -e
cd ~/NL_Integration_Project
module load GCCcore/10.3.0 Python/3.9.5 && . envs/py39/bin/activate
export PYTHONPATH=. DATA_DIR=$HOME/NL_Integration_Project/data
EMO=perlayer_emoctx_r0p5_w16:11
AVD=perlayer_emoctx_r0p5_w16_avd
OSM=opensmile_r0p5_w2
X=--exclude=gpu050

# one store per affective dimension -> emoctx16_{arousal,dominance,valence}
python3 scripts/split_avd_bands.py --store $AVD --out-prefix emoctx16

perm() { sbatch --parsable $X --dependency=aftercorr:$1 "${@:2}" scripts/cluster/permutation.sbatch; }

# ---- text x emotion, every context length (k512 = main model) -------------
for T in llama8b_k4L13 llama8b_k16L13 llama8b_k64L13 perlayer_llama8b_k256:13 llama8b_k512L13 perlayer_llama8b_k1024:13; do
  L=$(echo $T | grep -o 'k[0-9]*')
  J=$(sbatch --parsable $X --output=logs/encjoint_ctx${L}_%a_out --error=logs/encjoint_ctx${L}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$EMO scripts/cluster/encoding_joint.sbatch)
  C=$(perm $J --output=logs/perm_ctxc${L}_%a_out --error=logs/perm_ctxc${L}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$EMO,N_PERMS=1000,SHUFFLE=conditional,SAVE_NULL=1,TAGSUF=_null)
  M=$(perm $J --output=logs/perm_ctxm${L}_%a_out --error=logs/perm_ctxm${L}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$EMO,N_PERMS=1000,SHUFFLE=both,SAVE_NULL=1,TAGSUF=_marginalfixed)
  echo "$T x emotion: joint $J conditional $C marginalfixed $M"
done

# ---- text x eGeMAPS (k512) ------------------------------------------------
T=llama8b_k512L13
if [ -z "$SKIP_OSM" ]; then
J=$(sbatch --parsable $X --output=logs/encjoint_osm_%a_out --error=logs/encjoint_osm_%a_err \
    --export=ALL,TEXT=$T,AUDIO=$OSM scripts/cluster/encoding_joint.sbatch)
C=$(perm $J --output=logs/perm_osmc_%a_out --error=logs/perm_osmc_%a_err \
    --export=ALL,TEXT=$T,AUDIO=$OSM,N_PERMS=1000,SHUFFLE=conditional,SAVE_NULL=1,TAGSUF=_null)
M=$(perm $J --output=logs/perm_osmm_%a_out --error=logs/perm_osmm_%a_err \
    --export=ALL,TEXT=$T,AUDIO=$OSM,N_PERMS=1000,SHUFFLE=both,SAVE_NULL=1,TAGSUF=_marginalfixed)
echo "$T x eGeMAPS: joint $J conditional $C marginalfixed $M"
fi

# ---- text + AVD (accuracy table only) -------------------------------------
J=$(sbatch --parsable $X --output=logs/encjoint_tavd_%a_out --error=logs/encjoint_tavd_%a_err \
    --export=ALL,TEXT=$T,AUDIO=$AVD scripts/cluster/encoding_joint.sbatch)
echo "text + AVD: joint $J"

# ---- named-band fits: Draper-Stoneman null per band (each) ----------------
bands() {  # $1 log tag, $2 BANDS, $3 MODELS (may be empty), $4 marginalfixed too?
  local F E M=""
  F=$(sbatch --parsable $X --output=logs/encbands_$1_%a_out --error=logs/encbands_$1_%a_err \
      --export=ALL,BANDS="$2",MODELS="$3",TAG=ctx scripts/cluster/encoding_bands.sbatch)
  local D=results/encoding/bands_$(for b in $2; do printf '%s-' "${b%%=*}"; done | sed 's/-$//')__ctx
  E=$(perm $F --output=logs/perm_$1e_%a_out --error=logs/perm_$1e_%a_err \
      --export=ALL,BANDS="$2",MODELS="$3",SHUFFLE=each,FROM_DIR=$D,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_ctxnull)
  if [ "$4" = 1 ]; then
    M=$(perm $F --output=logs/perm_$1m_%a_out --error=logs/perm_$1m_%a_err \
        --export=ALL,BANDS="$2",MODELS="$3",SHUFFLE=both,FROM_DIR=$D,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_ctxmarginalfixed)
  fi
  echo "$D: fit $F each $E marginalfixed ${M:--}"
}
bands emoosm "emotion=$EMO opensmile=$OSM" "" 0
bands osmavd "opensmile=$OSM avd=$AVD" "" 0
bands av "arousal=emoctx16_arousal valence=emoctx16_valence" "" 1
bands avd "arousal=emoctx16_arousal dominance=emoctx16_dominance valence=emoctx16_valence" \
  "arousal dominance valence dominance+valence arousal+dominance arousal+valence joint" 1
echo SUBMITTED
