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

    Sound-relative, i.e. the same clock the TextGrids and the text band use.
    TR *i* starts at ``2i - TR_START_TIME`` seconds into the wav, so the first
    few onsets are negative: the scanner ran for 10 s before the sound began.
    Callers must handle that (clamp to 0, or drop those TRs) -- they are real
    TRs with no audio, not an error.

    **This used to add `soundstarttime` back**, returning the *scanner* clock
    (0, 2, 4, ...) and then using it to index the wav. That put every audio
    feature 10 s -- five TRs -- later than the text band and the responses it
    is regressed against. Measured on UTS01/`wheretheressmoke`, against the
    mean BOLD of the 344 voxels with EV>0.2: the text band peaks at a lag of
    2 TRs (r=+0.51), a textbook HRF delay, while the audio band peaked at 7
    TRs (r=+0.20). With `--ndelays 4` offering lags of 1-4 TRs, the prosody
    band's real signal was outside the model's reach entirely.

    Two independent checks agreed. The trailing rows of every openSMILE store
    are identical, which only happens if the last windows run past the end of
    the wav -- and solving for the wav duration that implies gives a value
    consistent with the TextGrid for the scanner clock, but *shorter than the
    last spoken word* for the sound clock, in 4 of 5 stories. And the stores
    have no leading silence, which rules out the alternative explanation that
    the wavs themselves carry a 10 s lead-in.

    Everything extracted before this fix is on the old grid and must be
    re-extracted: all five audio bands and the eGeMAPS fine-tuning targets.
    The fine-tuned checkpoints themselves are unaffected -- input windows and
    targets were shifted together, so the mapping they learned is unchanged --
    and the text bands were never on this path.

    Raises
    ------
    KeyError
        If `story` has no TR timing, which means it cannot be aligned and
        should be skipped rather than silently mis-windowed.
    """
    if story not in trfiles:
        raise KeyError(f"No TR timing for story {story!r}")
    tr_info = trfiles[story][0]
    return tr_info.get_reltriggertimes()


def stories_with_timing(stories: List[str], trfiles: Dict[str, list]) -> List[str]:
    """Subset of `stories` that have TR timing, order preserved."""
    return [s for s in stories if s in trfiles]


def window_bounds(onset: float, sr: int, window_samples: int,
                  n_samples: int):
    """Sample range for the window starting at `onset` seconds, plus padding.

    Returns ``(lo, hi, pad_left, pad_right)`` where ``signal[lo:hi]`` padded by
    ``pad_left``/``pad_right`` zeros is the requested window, always exactly
    `window_samples` long.

    This exists because `tr_onsets` is sound-relative, so its first few onsets
    are NEGATIVE -- the scanner ran ~10 s before the sound started, and those
    TRs have no audio. `int(onset * sr)` is then negative, and a negative start
    index slices from the END of the array in both numpy and torch: the window
    would be filled with audio from the wrong end of the story, silently, with
    the right shape and no error. Clamp and pad instead.
    """
    start = int(round(onset * sr))
    end = start + window_samples
    lo = min(max(start, 0), n_samples)
    hi = min(max(end, 0), n_samples)
    # Capped at the window length: a window entirely before the sound needs
    # `window_samples` of padding, not `-start` of it. pad_right is then
    # whatever is left over, so the three pieces always sum to the window.
    pad_left = min(max(-start, 0), window_samples)
    pad_right = window_samples - (hi - lo) - pad_left
    return lo, hi, pad_left, max(0, pad_right)
