import numpy as np
import json
from os.path import join
from .stimulus_utils import load_textgrids, load_simulated_trfiles
from .dsutils import make_word_ds

def get_story_wordseqs(stories, data_dir):
    """Get word sequences for specified stories."""
    grids = load_textgrids(stories, data_dir)
    with open(join(data_dir, "ds003020/derivative/respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs 