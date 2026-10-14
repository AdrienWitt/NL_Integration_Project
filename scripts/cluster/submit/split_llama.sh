#!/bin/bash
module load GCCcore/10.3.0 Python/3.9.5 && . envs/py39/bin/activate
export PYTHONPATH=. DATA_DIR=$HOME/NL_Integration_Project/data
python3 scripts/split_surprisal_bands.py --ks 1024 --store-prefix surprisal_llama8b_k --pred-prefix surp_pred_llama8b_k --wordrate-name wordrate_llama8b
python3 - <<"PY"
import h5py, numpy as np, glob, os
bad=0; n=0
for f in sorted(glob.glob("data/features/wordrate_llama8b/*.hf5")):
    a=h5py.File(f)["data"][:]; b=h5py.File(f.replace("wordrate_llama8b","wordrate"))["data"][:]
    n+=1; bad+= (a.shape!=b.shape) or (not np.allclose(a,b,atol=1e-5))
print(f"wordrate llama vs gpt2: {n} stories, {bad} differ")
PY
