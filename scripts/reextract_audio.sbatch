#!/bin/bash
#SBATCH --job-name=reextract
#SBATCH --output=logs/reext_%a_out
#SBATCH --error=logs/reext_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-4
#
# ===========================================================================
# Re-extract every audio band after the tr_onsets fix (2026-09-09).
# ===========================================================================
#
#   sbatch scripts/reextract_audio.sbatch
#
# Why this exists: `tr_onsets` returned the SCANNER clock and used it to index
# the wav, so every audio window covered wav [2i, 2i+2) where it should have
# covered [2i-10, 2i-8) -- ten seconds, five TRs, later than the text band and
# the responses. Measured on UTS01/wheretheressmoke against the mean BOLD of
# the 344 voxels with EV>0.2, the text band peaks at a 2-TR lag and the audio
# band peaked at 7. With --ndelays 4 the prosody signal was unreachable, so
# every prosody number so far is a floor rather than an estimate.
#
# The five stores below are exactly the ones on disk before the fix, same
# models and same layer ranges, so the re-run is comparable row for row:
#
#   opensmile               88 eGeMAPS functionals
#   perlayer_base_emotion   audeering emotion model, blocks 0-11
#   perlayer_base_robust    wav2vec2-large-robust, blocks 0-23
#   perlayer_ft_emotion     fine-tuned emotion checkpoint, blocks 6-11
#   perlayer_ft_robust      fine-tuned robust checkpoint, blocks 12-23
#
# The old stores are moved to data/features/_misaligned_<date>/ rather than
# overwritten: a half-finished overwrite would leave a directory that is part
# old grid and part new, with nothing to distinguish the two.
#
# NOT re-extracted here: the text bands (they were always on the sound clock)
# and prosody/finetune_targets (needed only if the fine-tuning is redone; the
# checkpoints themselves are unaffected, since input windows and targets were
# shifted together).

set -e
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

LOG="${TMPDIR:-/tmp}/reext_${SLURM_ARRAY_TASK_ID}.log"
trap 'cp -f "$LOG" "logs/reext_${SLURM_ARRAY_TASK_ID}_full.log" 2>/dev/null || true' EXIT
exec > "$LOG" 2>&1

# envs/prosody does not exist and GCCcore/12.3.0 is not the toolchain the
# packages were built against: transformers lives in the user site-packages
# and only resolves its dependencies under Python/3.9.5. This is the same
# environment scripts/extract_semantic.sbatch uses, which is the one that has
# actually run successfully.
module load GCCcore/10.3.0 Python/3.9.5
. "$PROJECT/envs/py39/bin/activate"

export PYTHONUNBUFFERED=1
export DATA_DIR=$PROJECT/data
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1

FT=results/finetune
SEED=${SEED:-42}
LR=${LR:-3e-05}   # note: 3e-05, the form the checkpoint dirs actually use

NAMES=(opensmile perlayer_base_emotion perlayer_base_robust
       perlayer_ft_emotion perlayer_ft_robust)
OUT=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "=== task $SLURM_ARRAY_TASK_ID : $OUT ==="
python3 -c "from common.tr_alignment import load_trfiles, tr_onsets; \
t=load_trfiles(); o=tr_onsets('adollshouse', t); \
print('tr_onsets[:3] =', o[:3]); \
assert o[0] < 0, 'tr_onsets is not on the sound clock -- is the fix pulled?'"

case "$OUT" in
  opensmile)
      python3 -m extract.opensmile --out-name opensmile ;;
  perlayer_base_emotion)
      python3 -m extract.wav2vec --model emotion --layers 0-11 --per-layer \
              --out-name perlayer_base_emotion ;;
  perlayer_base_robust)
      python3 -m extract.wav2vec --model wav2vec2-robust --layers 0-23 \
              --per-layer --out-name perlayer_base_robust ;;
  perlayer_ft_emotion)
      python3 -m extract.wav2vec \
              --model "$FT/emotion_frozen_6_lr${LR}_seed${SEED}/final_model" \
              --layers 6-11 --per-layer --out-name perlayer_ft_emotion ;;
  perlayer_ft_robust)
      python3 -m extract.wav2vec \
              --model "$FT/wav2vec2_robust_frozen_12_lr${LR}_seed${SEED}/final_model" \
              --layers 12-23 --per-layer --out-name perlayer_ft_robust ;;
  *) echo "unknown store: $OUT" >&2; exit 1 ;;
esac

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
