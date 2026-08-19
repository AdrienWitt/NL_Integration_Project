"""
Extract the TR-aligned eGeMAPS audio band for the encoding models.

Same 88 `EGEMAPS_FEATURE_SET` functionals the fine-tuning predicts, but written
in the feature format the encoding stage reads: one ``<story>.hf5`` per story
with a ``data`` dataset of shape (n_TRs, 88).

This is the interpretable prosody baseline. Comparing it against the wav2vec2
band answers a question worth asking directly: does a learned representation
buy anything over hand-designed acoustic descriptors?

Usage
-----
    python -m extract.opensmile --out-name opensmile
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from config import (EGEMAPS_FEATURE_SET, EGEMAPS_N_FUNCTIONALS, FEATURES_DIR,
                    STIMULI_16K_DIR, WINDOW_SIZE_SEC, ensure_dirs)
from common.tr_alignment import load_trfiles, tr_onsets

log = logging.getLogger("extract.opensmile")


def load_smile():
    import opensmile
    return opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, EGEMAPS_FEATURE_SET),
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_story(story: str, audio_dir: Path, onsets: np.ndarray, smile):
    import librosa

    y, sr = librosa.load(str(audio_dir / f"{story}.wav"), sr=None, mono=True)
    window_samples = int(WINDOW_SIZE_SEC * sr)
    rows, names = [], None

    for onset in onsets:
        start = int(onset * sr)
        end = start + window_samples
        if start >= len(y):
            window = np.zeros(window_samples, dtype=np.float32)
        elif end <= len(y):
            window = y[start:end]
        else:
            window = np.pad(y[start:], (0, end - len(y)), mode="constant")

        feats = smile.process_signal(window, sr)
        if names is None:
            names = list(feats.columns)
        rows.append(feats.values.reshape(1, -1))

    return np.vstack(rows).astype(np.float32), names


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--audio-dir", default=str(STIMULI_16K_DIR))
    p.add_argument("--out-name", default="opensmile")
    p.add_argument("--stories", default=None,
                   help="comma-separated subset; default is every wav found")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    audio_dir = Path(args.audio_dir)
    out_dir = Path(FEATURES_DIR) / args.out_name
    ensure_dirs(out_dir)

    trfiles = load_trfiles()
    smile = load_smile()

    if args.stories:
        stories = [s.strip() for s in args.stories.split(",") if s.strip()]
    else:
        stories = sorted(p.stem for p in audio_dir.glob("*.wav"))

    log.info(f"{EGEMAPS_FEATURE_SET} functionals -> {out_dir}")
    log.info(f"{len(stories)} stories")

    done = skipped = 0
    feature_names = None
    for story in stories:
        out_path = out_dir / f"{story}.hf5"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if story not in trfiles:
            log.warning(f"  {story}: no TR timing, skipping")
            skipped += 1
            continue
        if not (audio_dir / f"{story}.wav").exists():
            log.warning(f"  {story}: wav missing, skipping")
            skipped += 1
            continue

        features, names = extract_story(story, audio_dir,
                                        tr_onsets(story, trfiles), smile)
        if feature_names is None:
            feature_names = names
            if len(names) != EGEMAPS_N_FUNCTIONALS:
                raise ValueError(
                    f"openSMILE returned {len(names)} features, expected "
                    f"{EGEMAPS_N_FUNCTIONALS} for {EGEMAPS_FEATURE_SET}"
                )
            log.info(f"  {len(names)} features: {names[:3]} ... {names[-1]}")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=features)
        log.info(f"  {story}: {features.shape} -> {out_path.name}")
        done += 1

    if feature_names:
        with open(out_dir / "feature_names.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(feature_names))

    log.info(f"Done: {done} written, {skipped} skipped.")


if __name__ == "__main__":
    main()
