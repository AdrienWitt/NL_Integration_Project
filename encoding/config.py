import os
from os.path import join, dirname

# Get the absolute path to the repository root
REPO_DIR = join(dirname(dirname(os.path.abspath(__file__))))
EM_DATA_DIR = join(REPO_DIR, 'em_data')
DATA_DIR = REPO_DIR  # This will point to the NL_Project directory

