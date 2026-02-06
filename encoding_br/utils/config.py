import os
from os.path import join, dirname

# Get the absolute path to the repository root
REPO_DIR = join(dirname(dirname(dirname(os.path.abspath(__file__)))))
DER_DIR = join(REPO_DIR, 'derivative')
DATA_DIR = join(REPO_DIR, 'ds003020/derivative/preprocessed_data')

