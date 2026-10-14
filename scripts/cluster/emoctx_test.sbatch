#!/bin/bash
#SBATCH --job-name=emoctx_test
#SBATCH --output=logs/emoctx_test_out
#SBATCH --error=logs/emoctx_test_err
#SBATCH --partition=shared-gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,VramPerGpu:16G
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=gpu050
#SBATCH --constraint="COMPUTE_CAPABILITY_7_5|COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
# 2026-10-14: does extract.emotion_context (R=2, W=0) reproduce perlayer_base_emotion? + timing of R=0.5, W=16.
set -e
cd ~/NL_Integration_Project
module load GCCcore/10.3.0 Python/3.9.5
. envs/py39/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH=.
S=wheretheressmoke
python -m extract.emotion_context --readout 2 --context 0 --stories $S --out-name _test_emoctx_r2_w0 --overwrite
python - <<PY
import h5py, numpy as np
a=h5py.File("data/features/perlayer_base_emotion/$S.hf5")["data"][:]
b=h5py.File("data/features/_test_emoctx_r2_w0/$S.hf5")["data"][:]
print("shapes", a.shape, b.shape)
print("max abs diff", float(np.abs(a-b).max()), "max |a|", float(np.abs(a).max()))
print("per-layer corr", [round(float(np.corrcoef(a[:,l].ravel(), b[:,l].ravel())[0,1]),6) for l in range(12)])
PY
time python -m extract.emotion_context --readout 0.5 --context 16 --stories $S --out-name _test_emoctx_r0p5_w16 --overwrite
python - <<PY
import h5py, numpy as np
b=h5py.File("data/features/_test_emoctx_r0p5_w16/$S.hf5")["data"][:]
v=h5py.File("data/features/_test_emoctx_r0p5_w16_avd/$S.hf5")["data"][:]
print("r0.5 w16", b.shape, v.shape, "finite", bool(np.isfinite(b).all()), "avd mean", v.mean(0).round(3))
PY
