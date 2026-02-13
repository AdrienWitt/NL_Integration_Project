import os
from utils.config import DATA_DIR, REPO_DIR
import h5py
import numpy as np

def get_response(stories, subject):
    """Get the subject"s fMRI response for stories."""
    subject_dir = os.path.join(DATA_DIR, subject)
    resp = {}
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        data = hf["data"][:]
        resp[story] = data
    return np.array(resp)


def load_encoding_scores(subject):
    subject_dir = os.path.join(REPO_DIR, "encoding/results/opensmile_all_stories", subject)
    scores_path = os.path.join(subject_dir, "fold_corrs.npy")
    scores = np.load(scores_path)
    return scores


    

results_fold = load_encoding_scores("UTS01")