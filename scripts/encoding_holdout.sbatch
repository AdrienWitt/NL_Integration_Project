#!/bin/bash
#SBATCH --job-name=enc_holdout
#SBATCH --output=logs/enc_%a_out
#SBATCH --error=logs/enc_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-44
#
# ===========================================================================
# The first real encoding run: held-out story, all stories, all subjects.
# ===========================================================================
#
# The sweep chose the layers on cross-validation inside the training stories
# and never touched `wheretheressmoke`. This is the one evaluation on it, so
# the numbers here are out-of-sample in the way the sweep's deliberately were
# not. Running more than one band against it is still a choice made *before*
# seeing the result — the five below were fixed by the sweep, not picked here.
#
#   base_emotion_L11      best mean layer in the sweep
#   base_emotion_9to11    its neighbourhood, averaged
#   base_robust_L18       most consistent layer (9/9 subjects over openSMILE)
#   base_robust_15to18    its neighbourhood, averaged
#   opensmile             the 88 eGeMAPS descriptors, unchanged reference
#
# Fine-tuned checkpoints are deliberately absent. The sweep showed fine-tuning
# on eGeMAPS made the emotion model worse at every layer above the freeze
# boundary, in 8 of 9 subjects; spending the held-out story on a control that
# already lost would be spending it for nothing.
#
# Why this one is slow where the sweep was fast
# ---------------------------------------------
# No --max-stories. The sweep capped at 40 to rank layers cheaply; a reported
# number should use everything the subject heard, up to 84 stories and ~29k
# training TRs. Kernel ridge is cubic in TRs, so this is roughly 8x the sweep's
# per-fit cost — but there is only one fit per task here, not thirteen, and no
# nested loop: --eval holdout selects alphas by CV inside the training set and
# then scores once on the held-out story.
#
# Memory follows the same jump: the kernel is (29k x 29k) float32 = 3.4 GB, and
# the solver keeps several working copies, so VramPerGpu:40G rather than the
# sweep's 24G. --mem=96G covers the host side, where the full-voxel response of
# the held-out story's 10 repeats is the peak.

set -e

# logs/ is a symlink to scratch; see scripts/prosody_sweep_all.sbatch for why.
mkdir -p logs

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
BANDS=(base_emotion_L11 base_emotion_9to11
       base_robust_L18 base_robust_15to18
       opensmile)

# Band-major: tasks 0-8 are the sweep's best band across all nine subjects, so
# a throttled or interrupted queue still yields one complete, reportable band.
NSUBJ=${#SUBJECTS[@]}
BAND=${BANDS[$((SLURM_ARRAY_TASK_ID / NSUBJ))]}
SUBJ=${SUBJECTS[$((SLURM_ARRAY_TASK_ID % NSUBJ))]}

RESP="data/ds003020/derivative/preprocessed_data/$SUBJ"
if [ ! -d "$RESP" ]; then
    echo "ERROR: no responses for $SUBJ at $RESP"
    exit 1
fi

echo "=== task $SLURM_ARRAY_TASK_ID : $SUBJ / $BAND ==="
echo "    stories available: $(ls "$RESP"/*.hf5 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --models audio only. The semantic band is a moving target (GPT-2 layer choice
# and context weighting are both open), so delta and preference wait for it to
# settle; conditioning on a provisional version would date these numbers.
# --text-features is still passed because run_encoding loads that band whatever
# models are asked for; the fit itself never sees those columns.
python -m encoding.run_encoding \
    --subjects "$SUBJ" \
    --text-features gpt2_mean \
    --audio-features "$BAND" \
    --models audio \
    --backend banded \
    --eval holdout \
    --n-splits 5 \
    --min-ev 0.1 \
    --tag "$SUBJ" \
    --himalaya-backend torch_cuda

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
