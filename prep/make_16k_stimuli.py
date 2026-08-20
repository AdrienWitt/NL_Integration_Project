"""
Resample the ds003020 stimuli to 16 kHz mono — what wav2vec2/HuBERT/WavLM expect.

Reads `<--in-dir>/<story>.wav` (ds003020 ships 44.1 kHz stereo) and writes
`<--out-dir>/<story>.wav` at `SAMPLING_RATE`, mono, PCM_16.

Why this exists
---------------
`finetune.dataset` and `extract.wav2vec` both downmix and resample on the fly,
so the raw stimuli would work unchanged. Converting once is still worth it: the
16 kHz mono copies are about a third of the size, which matters when moving
them to a cluster, and it takes the resample out of the per-window path.

Audio identity matters
----------------------
The eGeMAPS targets in `<FINETUNE_TARGET_DIR>/averaged/` were computed from a
specific copy of the stimuli. A training pair is (2 s window -> the 88
functionals of *that* window), so feeding the model a different rendering of
the same story — denoised, re-normalised, differently trimmed — silently
breaks the correspondence. `--check-against` compares durations against another
directory and refuses to continue on a mismatch.

Usage
-----
    python -m prep.make_16k_stimuli
    python -m prep.make_16k_stimuli --in-dir /mnt/e/NL/clean_nl_preproc/ds003020/stimuli \
        --check-against data/ds003020/stimuli
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from config import SAMPLING_RATE, STIMULI_16K_DIR, STIMULI_DIR, ensure_dirs

log = logging.getLogger("prep.stimuli16k")

#: Durations may differ by this much and still count as the same recording.
DURATION_TOL_SEC = 0.05


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in-dir", default=str(STIMULI_DIR),
                   help="source wavs at their native rate")
    p.add_argument("--out-dir", default=str(STIMULI_16K_DIR))
    p.add_argument("--check-against", default=None, metavar="DIR",
                   help="verify every duration matches this directory's copy "
                        "before writing anything; use it when the source is "
                        "not the copy the eGeMAPS targets were built from")
    p.add_argument("--stories", default=None,
                   help="comma-separated subset (default: every wav found)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    import librosa
    import soundfile as sf

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"{in_dir} is not a directory")
    ensure_dirs(out_dir)

    if args.stories:
        stories = [s.strip() for s in args.stories.split(",") if s.strip()]
    else:
        stories = sorted(f.stem for f in in_dir.glob("*.wav"))
    if not stories:
        raise FileNotFoundError(f"no .wav files under {in_dir}")

    log.info("=" * 68)
    log.info(f"Resampling {len(stories)} stimuli to {SAMPLING_RATE} Hz mono")
    log.info(f"  in  : {in_dir}")
    log.info(f"  out : {out_dir}")
    log.info("=" * 68)

    # -- optional identity check, before writing anything -------------------
    if args.check_against:
        ref_dir = Path(args.check_against)
        log.info(f"Checking durations against {ref_dir} ...")
        mismatches, unreadable = [], []
        for story in stories:
            ref = ref_dir / f"{story}.wav"
            if not ref.exists():
                continue
            try:
                a, b = sf.info(in_dir / f"{story}.wav"), sf.info(ref)
            except Exception as exc:                       # dehydrated / corrupt
                unreadable.append(f"{story} ({exc.__class__.__name__})")
                continue
            if abs(a.duration - b.duration) > DURATION_TOL_SEC:
                mismatches.append(f"{story}: {a.duration:.2f}s vs {b.duration:.2f}s")
        if unreadable:
            log.warning(f"  could not compare {len(unreadable)}: "
                        f"{unreadable[:5]}")
        if mismatches:
            raise SystemExit(
                f"{len(mismatches)} duration mismatch(es) — these are not the "
                f"same recordings the eGeMAPS targets were built from, so the "
                f"labels would not describe the audio:\n  "
                + "\n  ".join(mismatches[:10])
            )
        log.info(f"  {len(stories) - len(unreadable)} durations match")

    # -- convert ------------------------------------------------------------
    written, skipped, failed = 0, 0, []
    for story in stories:
        src, dst = in_dir / f"{story}.wav", out_dir / f"{story}.wav"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            y, _ = librosa.load(str(src), sr=SAMPLING_RATE, mono=True)
        except Exception as exc:
            # OneDrive placeholders surface here as OSError/LibsndfileError.
            failed.append(f"{story}: {exc.__class__.__name__}: {exc}")
            continue

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak == 0.0:
            failed.append(f"{story}: decoded to silence")
            continue

        # Downmixing and resampling can push the float signal past 1.0 (the
        # ds003020 stimuli routinely land near 1.07), which PCM_16 would clip.
        # Scaling costs nothing that survives downstream: the feature
        # extractors z-score every window, so absolute level is discarded
        # anyway.
        note = ""
        if peak > 1.0:
            y = y / peak * 0.999
            note = f", scaled from peak {peak:.3f} to avoid clipping"

        sf.write(str(dst), y, SAMPLING_RATE, subtype="PCM_16")
        written += 1
        log.info(f"  {story}: {len(y) / SAMPLING_RATE:.1f}s{note}")

    log.info(f"\nWrote {written}, skipped {skipped} already present, "
             f"{len(failed)} failed")
    if failed:
        for line in failed[:10]:
            log.error(f"  {line}")
        raise SystemExit(
            f"{len(failed)} file(s) could not be converted. If these are "
            f"OneDrive placeholders, mark the folder 'always keep on this "
            f"device' and wait for the download to finish."
        )


if __name__ == "__main__":
    main()
