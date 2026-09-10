#!/bin/bash
#SBATCH --job-name=innercv
#SBATCH --output=logs/innercv_%a_out
#SBATCH --error=logs/innercv_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:32G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-2
#
# ===========================================================================
# Does the inner CV size move the score? One band, two settings, three subjects.
# ===========================================================================
#
#   sbatch scripts/inner_cv_check.sbatch
#
# run_prosody_sweep bounds the alpha-selection loop to --n-splits; run_encoding
# leaves it at leave-one-story-out. Both are nested -- the inner search runs on
# story_ids[train_idx] only, so neither leaks the outer test fold into the
# hyperparameter -- and the difference is cost and selection variance, not
# validity. What is not known is whether it moves the number.
#
#   inner = LOSO   5 outer x 23 inner = 115 fits    run_encoding's default
#   inner = 5      5 outer x  5 inner =  25 fits    what every sweep used
#
# Why it matters: if the gap is below ~0.001 the final table can quote the
# sweep's layer scores directly and the runner asymmetry is a non-issue. If it
# is larger, the sweep's layer *choice* deserves rechecking under LOSO, since
# the argmax was picked under the cheap setting -- and that would be a real
# finding rather than bookkeeping.
#
# Three subjects, chosen to span mask size: UTS02 is one of the three dense
# subjects (11,254 voxels at EV>0.1), UTS04 and UTS07 are typical of the other
# six (1,706-4,092). A difference that only shows up at one mask size is a
# difference in estimator variance, which is exactly the axis in question.
#
# base_emotion:11 because it is the decided band, and 4096 columns is the
# expensive end -- if the gap is negligible there it is negligible for the 88-d
# and 3-d bands too.

set -e
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

SUBJECTS=(UTS02 UTS04 UTS07)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

run () {   # tag, extra args
    echo "########## $SUBJ | inner=$1 : t+$((SECONDS / 60)) min"
    python3 -m encoding.run_encoding \
        --subjects "$SUBJ" \
        --band emotion=perlayer_base_emotion:11 \
        --models emotion \
        --stories-json common_stories_all9.json \
        --backend banded \
        --eval cv \
        --min-ev 0.1 \
        --max-repeats 5 \
        --n-splits 5 \
        --tag "$1" ${2:+$2}
}

run innerLOSO
run inner5 "--alpha-n-splits 5"

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
