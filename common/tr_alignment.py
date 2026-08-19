"""
TR timing for the ds003020 stimuli.

Every feature in this project is sampled once per TR: a window of length
`WINDOW_SIZE_SEC` starting at each TR onset. `tr_onsets` is the single
function that decides where those windows start, so audio features, prosody
targets and brain responses stay on exactly the same grid.
"""

import json
from typing import Dict, List

import numpy as np

from config import RESPDICT_PATH, TR, TR_PAD, TR_START_TIME
from .ridge_utils.stimulus_utils import load_simulated_trfiles


def load_trfiles(respdict_path=None, tr: float = TR, pad: int = TR_PAD,
                 start_time: float = TR_START_TIME) -> Dict[str, list]:
    """Return {story: [TRFile]} simulated from the response-length dictionary."""
    if respdict_path is None:
        respdict_path = RESPDICT_PATH
    with open(respdict_path, "r") as f:
        respdict = json.load(f)
    return load_simulated_trfiles(respdict, tr=tr, pad=pad, start_time=start_time)


def tr_onsets(story: str, trfiles: Dict[str, list]) -> np.ndarray:
    """Onset time (seconds into the wav) of every TR of `story`.

    Raises
    ------
    KeyError
        If `story` has no TR timing, which means it cannot be aligned and
        should be skipped rather than silently mis-windowed.
    """
    if story not in trfiles:
        raise KeyError(f"No TR timing for story {story!r}")
    tr_info = trfiles[story][0]
    return tr_info.get_reltriggertimes() + tr_info.soundstarttime


def stories_with_timing(stories: List[str], trfiles: Dict[str, list]) -> List[str]:
    """Subset of `stories` that have TR timing, order preserved."""
    return [s for s in stories if s in trfiles]
