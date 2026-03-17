import os
from collections import defaultdict

PREPROC_DIR = r"E:\NL\clean_nl_preproc\ds003020\derivative\preprocessed_data"

subject_stories = defaultdict(set)

for subject in sorted(os.listdir(PREPROC_DIR)):
    subject_path = os.path.join(PREPROC_DIR, subject)

    for fname in sorted(os.listdir(subject_path)):
        print(f"  {subject} | {fname}")  # ← show raw filenames so we know the convention