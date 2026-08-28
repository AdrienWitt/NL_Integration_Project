#!/bin/sh
#SBATCH --job-name=prosody_sweep
#SBATCH --output=prosody_sweep_%a_out
#SBATCH --error=prosody_sweep_%a_err
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-8%3

# Layer sweep for the prosodic band, one array task per subject.
#
# Log names carry %a and nothing else: nine fixed filenames that the next run
# overwrites, rather than a new pair per job id. A single shared name would
# have all nine tasks interleaving into one file.
#
# --array=0-8%3 runs three subjects at a time. Raise the %3 to go wider if the
# queue is short; the tasks are fully independent.
#
# Sizing notes
# ------------
# --mem=48G is generous, not necessary. The responses are masked to the
# explainable-variance voxels as each story is read, so the response matrix is
# ~0.4 GB rather than the 18 GB it would be at all 81k voxels. Measured peak is
# closer to 12 GB, most of it the held-out story's 10 repeats.
#
# The real constraint is GPU memory: himalaya's torch_cuda backend builds an
# (n_TRs, n_TRs) kernel per band. At 40 training stories that is ~0.7 GB, at
# all 83 it is ~3.1 GB, plus solver copies. VramPerGpu:20G restricts this to
# the A100 nodes; drop it to 11G to also match the RTX 2080 Ti and P100 nodes
# if the A100 queue is long — the sweep fits there too.

set -e

module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SUBJECTS="UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09"
SUBJ=$(echo $SUBJECTS | cut -d' ' -f$((SLURM_ARRAY_TASK_ID + 1)))

# Fail loudly and early rather than after loading a store: a missing subject
# means the response transfer did not finish, which is worth knowing now.
RESP="data/ds003020/derivative/preprocessed_data/$SUBJ"
if [ ! -d "$RESP" ]; then
    echo "ERROR: no responses for $SUBJ at $RESP"
    echo "The rsync from the acquisition machine has not delivered this subject."
    exit 1
fi
NSTORIES=$(ls "$RESP"/*.hf5 2>/dev/null | wc -l)
echo "=== $SUBJ : $NSTORIES response files ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# How many training stories the sweep sees. A sweep needs the *ranking* of
# layers, not a publishable number, so a subset is deliberate: kernel ridge is
# quadratic in TRs, and 40 stories costs a quarter of what 83 would.
MAXSTORIES=40

# Coarse grid: every third layer, plus the top layer of each stack, which is
# where fine-tuning changed the representation most (corr 0.03 against base at
# robust L23) and is therefore the most likely to behave unlike its neighbours.
#
# The fine-tuned stores start at their freeze boundary on purpose: below it the
# weights are byte-identical to the base model, so those layers ARE the base
# profile and sweeping them twice would measure the same thing.
# eGeMAPS is scored alongside every sweep under identical folds and mask. It is
# the reference line the layer profile needs: the fine-tuned models were trained
# to predict exactly those 88 descriptors, so "does any layer beat opensmile"
# asks whether the learned representation is worth more than the features it
# was distilled from. A negative answer there is a real, publishable result.
run_sweep () {   # store, configs
    echo "########## $SUBJ / $1 ##########"
    python -m encoding.run_prosody_sweep \
        --subjects "$SUBJ" \
        --source "$1" \
        --configs "$2" \
        --tag "$SUBJ" \
        --max-stories "$MAXSTORIES" \
        --baseline-features opensmile \
        --n-splits 5 \
        --eval cv \
        --min-ev 0.1 \
        --himalaya-backend torch_cuda
}

# --tag "$SUBJ" is not cosmetic: the sweep summary lands at
# results/encoding/prosody_sweep/<eval>/<source>/sweep.csv, a path with no
# subject in it, so nine array tasks would race for one file and only the last
# to finish would survive. The per-configuration voxel maps were already
# per-subject; only the summary needed the tag.

run_sweep perlayer_base_robust  "0 3 6 9 12 15 18 21 23"
run_sweep perlayer_ft_robust    "12 14 16 18 20 22 23"
run_sweep perlayer_base_emotion "0 3 6 9 11"
run_sweep perlayer_ft_emotion   "6 7 8 9 10 11"

echo "=== $SUBJ done ==="
