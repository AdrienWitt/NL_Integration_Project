"""Word sequences aligned to TR times, built from the ds003020 TextGrids."""

import json

from .stimulus_utils import load_textgrids, load_simulated_trfiles
from .dsutils import make_word_ds, get_transcript


def get_story_wordseqs(stories, textgrid_dir=None, respdict_path=None):
    """Return {story: DataSequence} with word onsets aligned to TR times.

    Paths default to `config.TEXTGRID_DIR` / `config.RESPDICT_PATH`, so callers
    normally pass only `stories`.
    """
    from config import RESPDICT_PATH, TR, TR_START_TIME, TR_PAD

    if respdict_path is None:
        respdict_path = RESPDICT_PATH

    grids = load_textgrids(stories, textgrid_dir)

    with open(respdict_path, "r") as f:
        respdict = json.load(f)

    trfiles = load_simulated_trfiles(
        respdict, tr=TR, start_time=TR_START_TIME, pad=TR_PAD
    )
    return make_word_ds(grids, trfiles)


def get_story_grids(stories, textgrid_dir=None):
    """Return {story: transcript} for the requested stories."""
    return get_transcript(load_textgrids(stories, textgrid_dir))
