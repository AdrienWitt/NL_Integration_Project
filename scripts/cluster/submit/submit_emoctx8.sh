#!/bin/bash
# 2026-10-20. Every analysis of the paper that uses the emotion band, redone with
# 8 s of preceding audio instead of 16 (selection-rule check: 8 s is the shortest
# context within noise of the best, 8 -> 16 s = +0.00003, 4/9). Same recipes as
# submit_acoustic_ctx.sh; the text x eGeMAPS chain does not involve the emotion
# band and is not redone. Already on disk and skipped: the k512 x W8 joint fit and
# its conditional null (submit_audio_context_k512.sh). The partial k4/k16 x W8
# joints of the cancelled first submission are overwritten.
# Named-band fits get TAG=ctx8 and _ctx8* nulls so the W16 runs (ctx) stay intact.
set -e
cd ~/NL_Integration_Project
module load GCCcore/10.3.0 Python/3.9.5 && . envs/py39/bin/activate
export PYTHONPATH=. DATA_DIR=$HOME/NL_Integration_Project/data
EMO=perlayer_emoctx_r0p5_w8:11
AVD=perlayer_emoctx_r0p5_w8_avd
OSM=opensmile_r0p5_w2
X=--exclude=gpu050

# one store per affective dimension -> emoctx8_{arousal,dominance,valence}
python3 scripts/split_avd_bands.py --store $AVD --out-prefix emoctx8

perm() { sbatch --parsable $X ${1:+--dependency=aftercorr:$1} "${@:2}" scripts/cluster/permutation.sbatch; }

# ---- text x emotion, every context length (k512 = main model) -------------
for T in llama8b_k4L13 llama8b_k16L13 llama8b_k64L13 perlayer_llama8b_k256:13 llama8b_k512L13 perlayer_llama8b_k1024:13; do
  L=$(echo $T | grep -o 'k[0-9]*')
  if [ "$L" = k512 ]; then   # joint + conditional already done
    M=$(perm "" --output=logs/perm_e8m${L}_%a_out --error=logs/perm_e8m${L}_%a_err \
        --export=ALL,TEXT=$T,AUDIO=$EMO,N_PERMS=1000,SHUFFLE=both,SAVE_NULL=1,TAGSUF=_marginalfixed)
    echo "$T x emotion8: marginalfixed $M"
    continue
  fi
  J=$(sbatch --parsable $X --output=logs/encjoint_e8${L}_%a_out --error=logs/encjoint_e8${L}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$EMO scripts/cluster/encoding_joint.sbatch)
  C=$(perm $J --output=logs/perm_e8c${L}_%a_out --error=logs/perm_e8c${L}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$EMO,N_PERMS=1000,SHUFFLE=conditional,SAVE_NULL=1,TAGSUF=_null)
  M=$(perm $J --output=logs/perm_e8m${L}_%a_out --error=logs/perm_e8m${L}_%a_err \
      --export=ALL,TEXT=$T,AUDIO=$EMO,N_PERMS=1000,SHUFFLE=both,SAVE_NULL=1,TAGSUF=_marginalfixed)
  echo "$T x emotion8: joint $J conditional $C marginalfixed $M"
done

# ---- text + AVD (accuracy table only) -------------------------------------
T=llama8b_k512L13
J=$(sbatch --parsable $X --output=logs/encjoint_tavd8_%a_out --error=logs/encjoint_tavd8_%a_err \
    --export=ALL,TEXT=$T,AUDIO=$AVD scripts/cluster/encoding_joint.sbatch)
echo "text + AVD8: joint $J"

# ---- named-band fits: Draper-Stoneman null per band (each) ----------------
bands() {  # $1 log tag, $2 BANDS, $3 MODELS (may be empty), $4 marginalfixed too?
  local F E M=""
  F=$(sbatch --parsable $X --output=logs/encbands_$1_%a_out --error=logs/encbands_$1_%a_err \
      --export=ALL,BANDS="$2",MODELS="$3",TAG=ctx8 scripts/cluster/encoding_bands.sbatch)
  local D=results/encoding/bands_$(for b in $2; do printf '%s-' "${b%%=*}"; done | sed 's/-$//')__ctx8
  E=$(perm $F --output=logs/perm_$1e_%a_out --error=logs/perm_$1e_%a_err \
      --export=ALL,BANDS="$2",MODELS="$3",SHUFFLE=each,FROM_DIR=$D,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_ctx8null)
  if [ "$4" = 1 ]; then
    M=$(perm $F --output=logs/perm_$1m_%a_out --error=logs/perm_$1m_%a_err \
        --export=ALL,BANDS="$2",MODELS="$3",SHUFFLE=both,FROM_DIR=$D,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_ctx8marginalfixed)
  fi
  echo "$D: fit $F each $E marginalfixed ${M:--}"
}
bands emoosm8 "emotion=$EMO opensmile=$OSM" "" 0
bands osmavd8 "opensmile=$OSM avd=$AVD" "" 0
bands av8 "arousal=emoctx8_arousal valence=emoctx8_valence" "" 1
bands avd8 "arousal=emoctx8_arousal dominance=emoctx8_dominance valence=emoctx8_valence" \
  "arousal dominance valence dominance+valence arousal+dominance arousal+valence joint" 1
echo SUBMITTED
