"""
eGeMAPS band at a choice of temporal resolution and acoustic context.

Counterpart of `extract.emotion_context` for the 88 eGeMAPS functionals, so that
the hand-designed band gets the same chance as the learned one:

    --readout R   window of each step and step size, seconds. R = 2 reproduces
                  `extract.opensmile` (one 2-s window per TR); R = 0.5 tiles the
                  story with 0.5-s windows, resampled to the TR centres with the
                  text band's Lanczos filter, normalised by its weight sum.
    --context W   functionals are computed over [s - W, s + R]. Functionals have
                  no attention, so context can only mean a longer summary window
                  ending at the same time; it never reaches before the sound.

Functionals that are undefined on a short window (e.g. no voiced frame) come back
as NaN or inf from openSMILE; they are set to 0 and counted in the log.

    python -m extract.opensmile_context --readout 0.5 --context 0 \\
        --stories-json data/derivative/common_stories_all9.json
"""

import argparse
import json
import logging
import math
from pathlib import Path

import h5py
import numpy as np

from config import FEATURES_DIR, STIMULI_16K_DIR, TR, ensure_dirs
from common.tr_alignment import load_trfiles, tr_onsets, window_bounds
from common.ridge_utils.interpdata import lanczosinterp2D
from extract.emotion_context import readout_starts
from extract.opensmile import load_smile

log = logging.getLogger("extract.opensmile_context")


def extract_story(y, sr, starts, readout, context, smile):
    r_samp = int(round(readout * sr))
    rows = []
    for s in starts:
        lo, hi, pad_l, pad_r = window_bounds(s, sr, r_samp, len(y))
        seg = y[lo:hi]
        if pad_l or pad_r:
            seg = np.pad(seg, (pad_l, pad_r), mode="constant")
        if context > 0 and s > 0:
            c_lo = max(0, int(round((s - context) * sr)))
            seg = np.concatenate([y[c_lo:lo], seg])
        rows.append(smile.process_signal(seg, sr).values.reshape(-1))
    return np.vstack(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readout", type=float, required=True)
    p.add_argument("--context", type=float, required=True)
    p.add_argument("--audio-dir", default=str(STIMULI_16K_DIR))
    p.add_argument("--stories-json", default=None)
    p.add_argument("--stories", default=None)
    p.add_argument("--out-name", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    import librosa

    r_tag = str(a.readout).replace(".", "p").rstrip("0").rstrip("p")
    out_dir = Path(FEATURES_DIR) / (a.out_name or f"opensmile_r{r_tag}_w{a.context:g}")
    ensure_dirs(out_dir)
    smile, trfiles, audio_dir = load_smile(), load_trfiles(), Path(a.audio_dir)
    if a.stories:
        stories = [s.strip() for s in a.stories.split(",") if s.strip()]
    elif a.stories_json:
        stories = json.load(open(a.stories_json))["common_stories"]
    else:
        stories = sorted(q.stem for q in audio_dir.glob("*.wav"))
    log.info(f"readout {a.readout} s, context {a.context} s, {len(stories)} stories -> {out_dir.name}")

    for story in stories:
        out_path = out_dir / f"{story}.hf5"
        if out_path.exists() and not a.overwrite:
            continue
        y, sr = librosa.load(str(audio_dir / f"{story}.wav"), sr=None, mono=True)
        onsets = tr_onsets(story, trfiles)
        starts = readout_starts(onsets, a.readout)
        X = extract_story(y, sr, starts, a.readout, a.context, smile)
        bad = ~np.isfinite(X)
        if bad.any():
            log.info(f"  {story}: {int(bad.sum())} undefined values "
                     f"({bad.any(0).sum()} features) set to 0")
            X[bad] = 0.0
        if not math.isclose(a.readout, TR):
            centres, tr_times = starts + a.readout / 2, onsets + TR / 2
            gain = lanczosinterp2D(np.ones((len(starts), 1)), centres, tr_times, window=3)
            X = lanczosinterp2D(X, centres, tr_times, window=3) / gain
        with h5py.File(out_path, "w") as f:
            d = f.create_dataset("data", data=X.astype(np.float32))
            d.attrs["readout_sec"] = float(a.readout)
            d.attrs["context_sec"] = float(a.context)
        log.info(f"  {story}: {len(starts)} steps -> {X.shape}")
    log.info("done")


if __name__ == "__main__":
    main()
