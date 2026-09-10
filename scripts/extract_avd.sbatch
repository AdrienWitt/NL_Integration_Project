#!/bin/bash
#SBATCH --job-name=avd
#SBATCH --output=logs/avd_out
#SBATCH --error=logs/avd_err
#SBATCH --partition=shared-gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# ===========================================================================
# The 3-d affective band: arousal, dominance, valence per TR.
# ===========================================================================
#
#   sbatch scripts/extract_avd.sbatch
#
# Same model the encoding already uses -- audeering/wav2vec2-large-robust-12-
# ft-emotion-msp-dim -- but read at the head instead of the hidden states. Its
# forward returns both: the 1024-d mean-pooled state (which is roughly
# perlayer_base_emotion:11, already on disk) and the 3 numbers the regression
# head produces from it. This job takes the second.
#
# Why it is worth a job of its own
# --------------------------------
# The audio band is 1024 opaque dimensions, so "this voxel is predicted by the
# audio band" says nothing about *what* it tracks. The head's three outputs are
# the same representation squeezed through a bottleneck that keeps only affect,
# and they have names. Comparing r(3-d) with r(1024-d) asks how much of the
# audio effect survives an affect-only bottleneck -- the only measurement in
# this project that addresses *affective* prosody rather than prosody.
#
# Do NOT use --output both. The 3 values are a function of the 1024-d vector,
# so concatenating them returns an opaque band and throws away the one thing
# that made this worth extracting: three columns you can name in a sentence.
#
# Order is arousal, dominance, valence -- id2label confirms it, and it is NOT
# the A/V/D order an older irony analysis may have assumed.
#
# The split step is where the encoding bands come from: run_encoding selects
# stores, not columns, so three bands means three directories. It also prints
# the correlation matrix, which decides whether "where is dominance vs arousal"
# is a question this model can answer at all (on ../Clean_Irony's 774 stimuli
# those two correlate 0.950; confirm it here before reading any per-dimension
# map).

set -e

# logs/ is a symlink to scratch; see scripts/prosody_sweep_all.sbatch for why.
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

module load GCCcore/10.3.0 Python/3.9.5
. "$PROJECT/envs/py39/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export DATA_DIR=$PROJECT/data

python3 -m extract.emotion_avd --output avd --out-name emotion_avd

echo
echo "=== splitting into one store per dimension ==="
python3 scripts/split_avd_bands.py --trim 5

echo "=== done in $((SECONDS / 60)) min ==="
