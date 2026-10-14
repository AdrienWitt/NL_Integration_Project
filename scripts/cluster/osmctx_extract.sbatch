#!/bin/bash
#SBATCH --job-name=osmctx_ext
#SBATCH --output=logs/osmctx_ext_%a_out
#SBATCH --error=logs/osmctx_ext_%a_err
#SBATCH --partition=shared-cpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-9
# 2026-10-14. eGeMAPS at readout {2, 0.5} s x context {0,2,4,8,16} s (same grid as
# emoctx). Task 0 first checks that R=2/W=0 reproduces the paper store `opensmile`.
set -e
cd ~/NL_Integration_Project
module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH=.
RS=(2 2 2 2 2 0.5 0.5 0.5 0.5 0.5); WS=(0 2 4 8 16 0 2 4 8 16)
R=${RS[$SLURM_ARRAY_TASK_ID]}; W=${WS[$SLURM_ARRAY_TASK_ID]}
python -m extract.opensmile_context --readout $R --context $W --stories-json data/derivative/common_stories_all9.json
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then python - <<PY
import h5py, numpy as np
a=h5py.File("data/features/opensmile/wheretheressmoke.hf5")["data"][:]
b=h5py.File("data/features/opensmile_r2_w0/wheretheressmoke.hf5")["data"][:]
print("REPRO", a.shape, b.shape, "max abs diff", float(np.nanmax(np.abs(a-b))))
PY
fi
echo "=== done in $((SECONDS / 60)) min ==="
