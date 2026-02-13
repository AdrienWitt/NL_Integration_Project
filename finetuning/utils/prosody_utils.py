import os
import sys
import json
from typing import Any, Dict

from .config import REPO_DIR

try:
    # Ensure project repo is on path for custom imports used elsewhere
    if REPO_DIR not in sys.path:
        sys.path.append(REPO_DIR)
except Exception:
    pass

SAMPLING_RATE = 16000
WINDOW_SIZE_SEC = 2.0
RESPDICT_PATH = os.path.join(REPO_DIR, "ds003020/derivative/respdict.json")


def load_trfiles(respdict_path: str = RESPDICT_PATH, tr: float = 2.0, pad: int = 5, start_time: int = 10):
    """Load simulated TR files using the project's stimulus utilities.

    This wraps the import so that the caller doesn't need to manage sys.path.
    """
    from encoding.utils.ridge_utils.stimulus_utils import load_simulated_trfiles

    with open(respdict_path, "r") as f:
        respdict = json.load(f)

    return load_simulated_trfiles(respdict, tr=tr, pad=pad, start_time=start_time)
