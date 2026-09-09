#!/bin/bash
#SBATCH --job-name=targets
#SBATCH --output=logs/targets_out
#SBATCH --error=logs/targets_err
#SBATCH --partition=shared-cpu
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# ===========================================================================
# Rebuild the eGeMAPS fine-tuning targets on the corrected window grid.
# ===========================================================================
#
#   sbatch scripts/make_targets.sbatch
#
# Why: `tr_onsets` returned the SCANNER clock until 2026-09-09, so the targets
# written on 2026-08-20 describe the audio of wav window [2i, 2i+2) where the
# corrected grid cuts [2i-10, 2i-8). Measured against the two openSMILE stores
# on the same [TR_PAD+trim : -trim] slice, the stored targets correlate 0.52-0.64
# with the misaligned store and 0.04-0.11 with the re-extracted one: they are on
# the old grid, unambiguously.
#
# The two existing checkpoints are NOT invalidated by this -- prep and
# finetune/dataset.py both called the same tr_onsets, so window and label were
# wrong together and the mapping they learned is intact. What breaks is any
# NEW fine-tuning run: dataset.py now cuts on the sound clock while these
# targets are on the scanner clock, a 10 s mismatch on every window, silent.
#
# --audio-dir is not optional. The script defaults to STIMULI_DIR, the
# native-rate wavs, and that directory does not exist on this cluster; only the
# 16 kHz copies do. Using them is also the more correct choice: the model is fed
# 16 kHz, so functionals describing content above 8 kHz cannot be predicted from
# what it hears. That difference is why the old targets only reach r=0.56
# against the 16 kHz store even on their own grid.
#
# CPU only -- openSMILE does not use the GPU.

set -e
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

LOG="${TMPDIR:-/tmp}/targets.log"
trap 'cp -f "$LOG" logs/targets_full.log 2>/dev/null || true' EXIT
exec > "$LOG" 2>&1

module load GCCcore/10.3.0 Python/3.9.5
. "$PROJECT/envs/py39/bin/activate"

export PYTHONUNBUFFERED=1
export DATA_DIR=$PROJECT/data

# Fail before spending an hour if the fix is not actually in this checkout.
python3 -c "from common.tr_alignment import load_trfiles, tr_onsets; \
o = tr_onsets('adollshouse', load_trfiles()); \
print('tr_onsets[:3] =', o[:3]); \
assert o[0] < 0, 'tr_onsets is not on the sound clock -- is the fix pulled?'"

python3 -m prep.make_finetune_targets \
    --trim 5 \
    --audio-dir "$PROJECT/data/stimuli_16k" \
    --overwrite

echo "=== done in $((SECONDS / 60)) min ==="
