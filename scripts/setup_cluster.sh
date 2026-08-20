#!/bin/sh
# One-time setup on the Baobab LOGIN node (compute nodes have no internet).
#   sh scripts/setup_cluster.sh
set -e

PROJECT=$HOME/NL_Integration_Project
cd "$PROJECT"

module load GCCcore/12.3.0 Python/3.11.3

python3 -m venv envs/prosody
. envs/prosody/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Prime the Hugging Face cache here — HF_HUB_OFFLINE=1 in the job means
# from_pretrained must never need the network.
export HF_HOME=$HOME/.cache/huggingface
python3 - <<'PY'
from transformers import AutoModel, AutoFeatureExtractor
for m in ["facebook/wav2vec2-large-robust",
          "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"]:
    print("fetching", m)
    AutoModel.from_pretrained(m)
    AutoFeatureExtractor.from_pretrained(m)
print("cache primed")
PY

python3 - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda build:", torch.version.cuda)
PY

echo
echo "Now check the data is in place:"
python3 -m config
