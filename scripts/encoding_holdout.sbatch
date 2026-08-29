#!/bin/bash
#SBATCH --job-name=enc_holdout
#SBATCH --output=logs/enc_%a_out
#SBATCH --error=logs/enc_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
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
# Memory follows the same jump, and 40G is NOT enough for the 83-story subjects:
# at n=27,797 training TRs himalaya asked for 14.4 GB on top of 33.2 GB already
# held and died on a 40 GB A100. UTS01-UTS03 need VramPerGpu:80G; every other
# subject fits in 40G because they heard a third as many stories. Override on
# the command line for those three rather than sending all 45 tasks to the nine
# 80 GB cards.
#
# --mem=96G covers the host side, where the full-voxel response of the held-out
# story's 10 repeats is the peak.
#
# The --constraint above is not about capacity. torch 2.6+cu124 ships no sm_120
# kernels, so a task landing on an RTX PRO 4500 Blackwell dies with "no kernel
# image is available for execution on the device" -- the card is too new for
# the wheel, not too old. Four c25 tasks were lost to it. The whitelist names
# the architectures this build actually has kernels for.

set -e

# logs/ is a symlink to scratch; see scripts/prosody_sweep_all.sbatch for why.
mkdir -p logs

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# himalaya allocates the kernel, its eigenvectors and a per-alpha working copy
# in quick succession, which fragments the pool badly enough to fail an
# allocation while enough total memory is free. Costs nothing when it is not
# needed.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
BANDS=(base_emotion_L11 base_emotion_9to11
       base_robust_L18 base_robust_15to18
       opensmile)

# Band-major: tasks 0-8 are the sweep's best band across all nine subjects, so
# a throttled or interrupted queue still yields one complete, reportable band.
NSUBJ=${#SUBJECTS[@]}
BAND=${BANDS[$((SLURM_ARRAY_TASK_ID / NSUBJ))]}
SUBJ=${SUBJECTS[$((SLURM_ARRAY_TASK_ID % NSUBJ))]}

# Which story list each subject trains on. The default, all_stories.json, gives
# every subject everything they heard -- 83 training stories for UTS01-03 but
# only 25 for UTS04, a 3.2x spread. That is the right choice for the band
# contrast, which is entirely within subject: same subject, same stories, same
# folds, so training-set size cancels exactly.
#
# It is the wrong choice for anything cross-subject. A group mean over these
# would be part prosody and part "UTS01 heard three times more data", so any
# group-level number needs common_stories_25.json instead, where all nine sit
# on the same 24-25 stories. Run both; they answer different questions.
#
#   STORIES_JSON=common_stories_25.json TAG_SUFFIX=_c25 \
#       sbatch scripts/encoding_holdout.sbatch
STORIES_JSON=${STORIES_JSON:-all_stories.json}
TAG_SUFFIX=${TAG_SUFFIX:-}

RESP="data/ds003020/derivative/preprocessed_data/$SUBJ"
if [ ! -d "$RESP" ]; then
    echo "ERROR: no responses for $SUBJ at $RESP"
    exit 1
fi

echo "=== task $SLURM_ARRAY_TASK_ID : $SUBJ / $BAND ==="
echo "    story list: $STORIES_JSON"
echo "    stories on disk: $(ls "$RESP"/*.hf5 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --models audio only. The semantic band is a moving target (GPT-2 layer choice
# and context weighting are both open), so delta and preference wait for it to
# settle; conditioning on a provisional version would date these numbers.
# --text-features is still passed because run_encoding loads that band whatever
# models are asked for; the fit itself never sees those columns.
python -m encoding.run_encoding \
    --subjects "$SUBJ" \
    --stories-json "$STORIES_JSON" \
    --text-features gpt2_mean \
    --audio-features "$BAND" \
    --models audio \
    --backend banded \
    --eval holdout \
    --n-splits 5 \
    --min-ev 0.1 \
    --tag "${SUBJ}${TAG_SUFFIX}" \
    --himalaya-backend torch_cuda

echo "=== task $SLURM_ARRAY_TASK_ID done in $((SECONDS / 60)) min ==="
