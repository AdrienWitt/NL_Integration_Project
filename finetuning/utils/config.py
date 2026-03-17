import os
from os.path import join, dirname

# Get the absolute path to the repository root
REPO_DIR = join(dirname(dirname(dirname(os.path.abspath(__file__)))))
MAIN_DIR = join(REPO_DIR, "finetuning")
STIMULI_DIR = join(REPO_DIR, 'stimuli_16k')
PROSODY_DIR = join(REPO_DIR, 'features/prosody/opensmile')
DATA_DIR = join(REPO_DIR, 'ds003020/derivative/preprocessed_data')
DER_DIR = join(REPO_DIR, 'derivative')
PROSODY_BRAIN_DIR = join(REPO_DIR, 'features/prosody/brain_targets_finetuning')



