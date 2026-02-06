import os
from os.path import join, dirname

# Get the absolute path to the repository root
REPO_DIR = join(dirname(dirname(dirname(os.path.abspath(__file__)))))
STIMULI_DIR = join(REPO_DIR, 'stimuli_16k')
PROSODY_DIR = join(REPO_DIR, 'features/prosody/opensmile')

