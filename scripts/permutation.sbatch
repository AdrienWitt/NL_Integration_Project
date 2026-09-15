#!/bin/bash
#SBATCH --job-name=perm
#SBATCH --output=logs/perm_%a_out
#SBATCH --error=logs/perm_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --array=0-8
#
# ===========================================================================
# The inference: Draper-Stoneman conditional nulls for delta and preference.
# ===========================================================================
#
#   # size it first -- see "Run the probe" below
#   sbatch --export=ALL,N_PERMS=20,TAGSUF=_probe scripts/permutation.sbatch
#
#   # then the real thing
#   sbatch --export=ALL,N_PERMS=1000 scripts/permutation.sbatch
#
# Reads the joint fit written by `scripts/encoding_joint.sbatch` and tests it.
# Nothing here re-searches alphas: `--from-encoding` loads that run's band
# weights, its three correlation maps and its voxel mask, so the observed
# statistic is literally the published one rather than a second fit that
# agrees with it. `load_observed_fit` refuses on any design mismatch --
# including, since this job was written, a different training-story set, which
# every other check would have let through.
#
# What the two nulls are
# ----------------------
# `--shuffle-block conditional` runs both directions and their conjunction:
#
#   delta_audio_given_text   audio's rows block-shuffled in the training
#                            design, text left aligned, joint model refit with
#                            its observed band weights. H0: "audio adds nothing
#                            beyond text", with the nesting advantage priced in
#                            because the shuffled columns are still there to
#                            overfit with.
#   delta_text_given_audio   the mirror.
#   delta                    the conjunction, p = max of the two.
#
# The conjunction is exact rather than approximate:
#
#   r_joint - max(r_text, r_audio) == min(r_joint - r_text, r_joint - r_audio)
#
# so `delta > 0` holds iff both conditional contributions are positive, and the
# pair is an intersection-union test of H0: delta <= 0 -- valid at level alpha
# with no correction between the two components (Berger 1982), and
# conservative, which is the right direction for a claim about integration.
#
# Why NOT --shuffle-block both
# ----------------------------
# That null shuffles every band, i.e. H0 "no encoding anywhere", under which
# all three correlations collapse to ~0 while the observed delta is a
# difference of two *large* correlations that banded ridge makes positive
# almost by construction. A purely semantic voxel clears it and gets labelled
# integrative. run_permutation writes it as `delta_naive_*` so it cannot be
# mistaken for the real test, and refuses it outright under --from-encoding
# (it needs test-set predictions run_encoding does not save).
#
# Run the probe
# -------------
# Nobody has measured what a permutation costs at this design, and the honest
# answer is that it is subject-dependent by an order of magnitude: the DS refit
# is a conjugate-gradient solve over the selected voxels, and the nine masks
# run from 1,706 (UTS04) to 16,395 (UTS03). 1,000 permutations x 2 directions
# on the largest mask could plausibly exceed the 12 h partition limit while the
# smallest finishes in under an hour. A timed-out task writes nothing.
#
# So: run N_PERMS=20 first, read the per-100 progress lines out of the logs,
# and size the real submission from the slowest subject. `progress_every` is
# n_perms/10, so a 20-permutation probe logs every 2.
#
# blocklen=10
# -----------
# 10 TRs = 20 s, comfortably past the HRF. A free row shuffle would destroy the
# temporal autocorrelation of continuous listening, narrow the null and make
# the test anticonservative. Calibration was measured on synthetic data with
# known ground truth -- H0-true voxels reject at 5.0% against a nominal 5%,
# `stats/validate_null.py`.
#
# n_perms=1000
# ------------
# The smallest attainable p is 1/(n_perms+1), so 1,000 permutations floor the
# per-voxel p at 0.000999. That is well below what FDR needs across a ~2,000 to
# ~16,000 voxel ROI, and it is a *within-subject* test with 8,683 training TRs
# -- not the n=9 group test, whose floor is 1/512 and which this file argues is
# arithmetically hopeless whole-brain. The group claim is assembled afterwards
# from these nine maps, as consistency or as max-statistic sign-flip FWE.

set -e

# Run marker. --output/--error use fixed names by project convention, so this
# file still holds the PREVIOUS run's text between submission and dispatch.
#     sed -n "/=== RUN $SLURM_JOB_ID /,\$p" logs/<file>
echo "=== RUN ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} started $(date -Is) on $(hostname) ==="
echo "=== RUN ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} started $(date -Is) on $(hostname) ===" >&2
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

TEXT=${TEXT:-perlayer_gpt2_k16:8}
AUDIO=${AUDIO:-perlayer_base_emotion:11}
N_PERMS=${N_PERMS:-1000}
BLOCKLEN=${BLOCKLEN:-10}
SHUFFLE=${SHUFFLE:-conditional}
STORIES_JSON=${STORIES_JSON:-common_stories_all9.json}
TAGSUF=${TAGSUF:-}

# The encoding run being tested. Built the same way run_encoding builds it --
# `store:layer` with the colon turned into an L -- so the two stay in step if
# the bands ever change.
FROM=results/encoding/$(echo "$TEXT" | tr ':' 'L')__$(echo "$AUDIO" | tr ':' 'L')

if [ ! -d "$FROM/banded/holdout/$SUBJ" ]; then
    echo "ERROR: $FROM/banded/holdout/$SUBJ does not exist." >&2
    echo "Run scripts/encoding_joint.sbatch first (--eval holdout)." >&2
    exit 1
fi

echo "=== $SUBJ | $SHUFFLE | $N_PERMS perms | blocklen $BLOCKLEN ==="
echo "=== observed fit: $FROM ==="

python3 -m stats.run_permutation \
    --subjects "$SUBJ" \
    --text-features "$TEXT" \
    --audio-features "$AUDIO" \
    --stories-json "$STORIES_JSON" \
    --from-encoding "$FROM" \
    --shuffle-block "$SHUFFLE" \
    --n-perms "$N_PERMS" \
    --blocklen "$BLOCKLEN" \
    --min-ev 0.1 \
    --max-repeats 5 \
    ${TAGSUF:+--tag "${TAGSUF#_}"}

echo "=== $SUBJ done in $((SECONDS / 60)) min ==="
