import os
from os.path import join, dirname

# Get the absolute path to the repository root
REPO_DIR = join(dirname(dirname(dirname(os.path.abspath(__file__)))))
EM_DATA_DIR = join(REPO_DIR, 'Huth', 'em_data')
DATA_DIR = join(REPO_DIR, 'ds003020', 'derivative','preprocessed_data')
TEXT_EMB = join(REPO_DIR, 'features', 'gpt2')
AUDIO_EMB = join(REPO_DIR, 'features', 'wav2vec')
