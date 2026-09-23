#!/bin/bash
#SBATCH --job-name=bench_solvers
#SBATCH --output=logs/bench_solvers_out
#SBATCH --error=logs/bench_solvers_err
#SBATCH --partition=shared-gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:80G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Measures primal vs dual ridge at the sweep's real shapes. See the docstring
# in scripts/bench_solvers.py for why the question is worth an hour of GPU.

set -e

# logs/ is a symlink to scratch. Writing this job's stdout to $HOME is what
# killed its first run: /home is a BeeGFS volume at 97% full, and himalaya's
# progress bar writes to stdout thousands of times per fit.
mkdir -p logs
module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/bench_solvers.py
