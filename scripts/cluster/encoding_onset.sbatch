#!/bin/bash
#SBATCH --job-name=enc_onset
#SBATCH --output=logs/enconset_%a_out
#SBATCH --error=logs/enconset_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --constraint="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-8
#
# ===========================================================================
# Does long context still win once the story-onset clock has its own band?
# ===========================================================================
#
#   PYTHONPATH=. python3 scripts/make_onset_band.py      # once, CPU, seconds
#   sbatch --array=3 scripts/encoding_onset.sbatch       # probe, time it
#   sbatch scripts/encoding_onset.sbatch
#
# For each K in $KS, two bands and three models in one design:
#
#   text    perlayer_gpt2_k${K}:8
#   onset   8 smooth functions of time since story onset (make_onset_band.py)
#
#   models  joint (= text+onset), text, onset
#
# The k comparison is r(joint) - r(onset): what the text band adds once the
# onset ramp is already in the model. At k >= 256 the text band decodes time
# since onset at r = 0.82 over the first 40 TRs (k16: 0.09), and the voxels
# where text|audio jumps from k16 to k256 are the voxels with an onset ramp in
# the responses, 9/9 subjects. `text` alone is kept so the old k16-vs-k256 gap
# is reproduced inside the same run.

set -e

RUN_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
echo "=== RUN ${RUN_ID} started $(date -Is) on $(hostname) ==="
echo "=== RUN ${RUN_ID} started $(date -Is) on $(hostname) ===" >&2
mkdir -p logs

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

module load GCCcore/10.3.0 Python/3.9.5
. "$PROJECT/envs/py39/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export DATA_DIR=$PROJECT/data
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

KS=${KS:-"16 256"}
# The text band per K, with {K} substituted, and the tag prefix. Defaults are
# the GPT-2 run; for Llama:
#   TEXT_FMT="perlayer_llama8b_k{K}:13" TAGPRE=llamaonset KS="16 256 1024"
# (k=16 is the flat store llama8b_k16L13, so pass it per K via TEXT_K16=...)
TEXT_FMT=${TEXT_FMT:-"perlayer_gpt2_k{K}:8"}
TAGPRE=${TAGPRE:-onset}
ONSET=${ONSET:-onset}
SEED=${SEED:-0}
EVALS=${EVALS:-"cv holdout"}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}

for K in $KS; do
  TEXT="${TEXT_FMT//\{K\}/$K}"
  OVR="TEXT_K${K}"
  if [ -n "${!OVR:-}" ]; then
      TEXT="${!OVR}"
  fi
  for EV in $EVALS; do
    echo "=== $SUBJ | k=$K | text=$TEXT | --eval $EV | $(date -Is) ==="
    # --n-splits is the OUTER fold count under cv; under holdout it IS the
    # alpha search, so it is passed only for cv.
    SPLITS=""
    [ "$EV" = "cv" ] && SPLITS="--n-splits 5"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        --band "text=${TEXT}" \
        --band "onset=${ONSET}" \
        --models joint text onset \
        --tag "${TAGPRE}k${K}" \
        --stories-json "$STORIES_JSON" \
        --backend banded \
        --eval "$EV" \
        --min-ev 0.1 \
        --seed "$SEED" \
        --max-repeats 5 \
        $SPLITS
  done
done

echo "=== RUN ${RUN_ID} $SUBJ done in $((SECONDS / 60)) min ==="
