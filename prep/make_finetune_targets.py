"""
Build the fine-tuning targets: 88 eGeMAPS functionals per TR, one JSON per story.

Output goes to ``<FINETUNE_TARGET_DIR>/averaged/``, which is what
`finetune.dataset.ProsodyDataset` reads.

Audio only
----------
This script used to also build brain targets: it masked fsaverage vertices to
the top 5% by encoding r, averaged responses across subjects, fitted one global
PCA, and wrote the components alongside the audio features for multi-task
fine-tuning. That path was removed on 2026-08-19 — training the encoder on
brain responses and then using its features for voxelwise encoding is circular,
and doubly so when the vertices were *selected by encoding r* in the first
place. See `trash/brain_pca_multitask/` for the previous version.

The practical consequence is that this script no longer needs `--corrs-dir`,
`--fsaverage-dir`, or any fMRI data at all: stimulus wavs and the TR grid are
enough. It also no longer skips stories that happen to lack fsaverage
responses.

Windows
-------
One row per TR: the `WINDOW_SIZE_SEC` window starting at that TR's onset. The
rows are then trimmed to ``[TR_PAD + trim : -trim]`` so they sit on the same
grid `finetune.dataset` slices the onsets to. `--trim` must match the value
passed to `finetune.run_finetune`.

Usage
-----
    python -m prep.make_finetune_targets
    python -m prep.make_finetune_targets --stories adollshouse,itsabox
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from config import (EGEMAPS_FEATURE_SET, EGEMAPS_N_FUNCTIONALS,
                    FINETUNE_TARGET_DIR, STIMULI_DIR, TR, TR_PAD,
                    WINDOW_SIZE_SEC, ensure_dirs)
from common.tr_alignment import load_trfiles, tr_onsets

log = logging.getLogger("prep.targets")


def load_smile():
    import opensmile
    return opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, EGEMAPS_FEATURE_SET),
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_story_features(story: str, audio_dir: Path, onsets: np.ndarray,
                           smile):
    """(n_TRs, 88) eGeMAPS functionals, one row per TR window."""
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

    return np.vstack(rows), names


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--audio-dir", default=str(STIMULI_DIR),
                   help="directory of <story>.wav (native sample rate)")
    p.add_argument("--out", default=str(FINETUNE_TARGET_DIR))
    p.add_argument("--stories", default=None,
                   help="comma-separated subset; default is every wav that "
                        "has TR timing")
    p.add_argument("--trim", type=int, default=5,
                   help="must match the --trim used in fine-tuning")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild stories whose JSON already exists")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out)
    avg_dir = out_dir / "averaged"
    ensure_dirs(out_dir, avg_dir)

    trfiles = load_trfiles(tr=TR, pad=TR_PAD)
    offset = TR_PAD + args.trim

    if args.stories:
        stories = [s.strip() for s in args.stories.split(",") if s.strip()]
    else:
        stories = sorted(p.stem for p in audio_dir.glob("*.wav"))

    usable, no_timing = [], []
    for story in stories:
        if not (audio_dir / f"{story}.wav").exists():
            raise FileNotFoundError(f"{audio_dir / (story + '.wav')} missing")
        (usable if story in trfiles else no_timing).append(story)

    log.info("=" * 68)
    log.info("Fine-tuning targets: eGeMAPS functionals per TR (audio only)")
    log.info(f"  feature set : {EGEMAPS_FEATURE_SET} "
             f"({EGEMAPS_N_FUNCTIONALS} functionals expected)")
    log.info(f"  window      : {WINDOW_SIZE_SEC:g} s from each TR onset")
    log.info(f"  trim        : [{offset}:-{args.trim}] TRs")
    log.info(f"  stories     : {len(usable)} usable, {len(no_timing)} without "
             f"TR timing")
    log.info("=" * 68)
    if no_timing:
        log.warning(f"  no TR timing, skipped: {no_timing}")

    smile = load_smile()
    feature_names = None
    written = 0

    for story in usable:
        safe = "".join(c if c.isalnum() or c in " _-" else "_"
                       for c in story).replace(" ", "_").strip("_")
        out_path = avg_dir / f"{safe}_prosody.json"
        if out_path.exists() and not args.overwrite:
            log.info(f"  {story}: exists, skipping (use --overwrite)")
            continue

        onsets = tr_onsets(story, trfiles)
        raw, names = extract_story_features(story, audio_dir, onsets, smile)

        if feature_names is None:
            feature_names = names
            log.info(f"  {len(names)} audio features "
                     f"(expected {EGEMAPS_N_FUNCTIONALS})")
            if len(names) != EGEMAPS_N_FUNCTIONALS:
                raise ValueError(
                    f"openSMILE returned {len(names)} features, not "
                    f"{EGEMAPS_N_FUNCTIONALS}. Check that "
                    f"{EGEMAPS_FEATURE_SET} Functionals is installed."
                )
        elif names != feature_names:
            raise ValueError(f"{story}: feature names differ from earlier stories")

        trimmed = raw[offset: -args.trim] if args.trim else raw[offset:]
        if trimmed.shape[0] <= 0:
            raise ValueError(
                f"{story}: {raw.shape[0]} TRs is too short for "
                f"[{offset}:-{args.trim}]"
            )

        payload = {
            "story": story,
            "n_TRs": int(trimmed.shape[0]),
            "tr_length_sec": TR,
            "audio_window_sec": WINDOW_SIZE_SEC,
            "feature_names": feature_names,
            "audio_features": {
                "tr_aligned": {
                    "description": f"{EGEMAPS_FEATURE_SET} functionals, window "
                                   f"starts at the TR onset; raw values "
                                   f"(z-scored in the dataset)",
                    "data": trimmed.tolist(),
                }
            },
            "metadata": {
                "audio_preproc": f"trimmed [{offset}:-{args.trim}] TRs",
                "trim": args.trim,
                "tr_pad": TR_PAD,
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        written += 1
        log.info(f"  {story}: {raw.shape[0]} -> {trimmed.shape[0]} TRs")

    log.info(f"\nDone: {written} stories written to {avg_dir}")


if __name__ == "__main__":
    main()
