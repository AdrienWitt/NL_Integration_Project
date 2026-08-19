"""
Split stories into train / val / held-out test for fine-tuning.

Rules
-----
* The repeated story (`HELD_OUT_STORY`) is removed from the pool entirely. It
  is the only story with enough repeats for an explainable-variance ceiling, so
  it is reserved for the final encoding test and must never be seen during
  fine-tuning — otherwise the encoding evaluation is contaminated by a model
  that has already heard the test audio.
* Stories every subject heard ("universal") all go to *train*. They are the
  most valuable for the encoding stage, where a story is only usable if every
  subject has it, so spending them on validation would waste them.
* Validation is drawn from the remaining non-universal stories.

Usage
-----
    python -m prep.make_story_splits
    python -m prep.make_story_splits --test-size 0.15 --seed 7
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import FMRI_DIR, HELD_OUT_STORY, SPLIT_DIR, ensure_dirs


def discover_subject_stories(fmri_dir: Path, prefix: str = "UTS") -> dict:
    """{subject: {stories}} from the response files on disk."""
    subject_stories = defaultdict(set)
    if not fmri_dir.is_dir():
        raise FileNotFoundError(
            f"{fmri_dir} not found. Set FMRI_DIR to the preprocessed responses."
        )
    for subject_dir in sorted(fmri_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith(prefix):
            continue
        for path in sorted(subject_dir.glob("*.hf5")):
            subject_stories[subject_dir.name].add(path.stem)
    if not subject_stories:
        raise FileNotFoundError(
            f"No '{prefix}*' subject folders with .hf5 files under {fmri_dir}"
        )
    return dict(subject_stories)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fmri-dir", default=str(FMRI_DIR))
    p.add_argument("--out", default=str(SPLIT_DIR))
    p.add_argument("--test-size", type=float, default=0.2,
                   help="fraction of non-universal stories used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--held-out", default=HELD_OUT_STORY)
    args = p.parse_args(argv)

    subject_stories = discover_subject_stories(Path(args.fmri_dir))
    print(f"Found {len(subject_stories)} subjects:")
    for subject, stories in sorted(subject_stories.items()):
        print(f"  {subject}: {len(stories)} stories")

    universal = sorted(set.intersection(*subject_stories.values()))
    all_heard = sorted(set.union(*subject_stories.values()))
    print(f"\nUniversal stories (heard by everyone): {len(universal)}")

    # Drop the held-out story and any repeat-suffixed variant of it.
    def is_held_out(story: str) -> bool:
        return story.startswith(args.held_out)

    available = [s for s in all_heard if not is_held_out(s)]
    universal_available = [s for s in universal if not is_held_out(s)]
    non_universal = [s for s in available if s not in set(universal_available)]

    print(f"Held out           : {args.held_out} "
          f"({'universal' if args.held_out in universal else 'NOT universal'})")
    print(f"Universal available: {len(universal_available)}")
    print(f"Non-universal      : {len(non_universal)}")

    if len(non_universal) < 2:
        raise RuntimeError(
            f"Only {len(non_universal)} non-universal stories — not enough to "
            f"build a validation set. Add subjects, or split the universal "
            f"stories instead."
        )

    extra_train, val_stories = train_test_split(
        non_universal, test_size=args.test_size, random_state=args.seed
    )

    train_stories = sorted(set(universal_available) | set(extra_train))
    val_stories = sorted(val_stories)
    test_stories = sorted(s for s in all_heard if is_held_out(s))

    assert not set(train_stories) & set(val_stories), "train/val overlap"
    assert not set(train_stories) & set(test_stories), "train/test overlap"
    assert not set(val_stories) & set(test_stories), "val/test overlap"

    print(f"\n{'=' * 60}")
    print(f"train : {len(train_stories)}")
    print(f"val   : {len(val_stories)} -> {val_stories}")
    print(f"test  : {len(test_stories)} -> {test_stories}")
    print(f"{'=' * 60}")

    print(f"\n{'Subject':10s} {'Total':>6s} {'Train':>7s} {'Val':>6s} {'Test':>6s}")
    print("  " + "-" * 38)
    for subject, stories in sorted(subject_stories.items()):
        print(f"{subject:10s} {len(stories):>6d} "
              f"{len(stories & set(train_stories)):>7d} "
              f"{len(stories & set(val_stories)):>6d} "
              f"{len(stories & set(test_stories)):>6d}")

    ensure_dirs(args.out)
    out_path = Path(args.out) / "stories_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "train": train_stories,
            "val": val_stories,
            "test": test_stories,
            "n_train": len(train_stories),
            "n_val": len(val_stories),
            "n_test": len(test_stories),
            "universal_stories": universal_available,
            "held_out_test": test_stories,
            "test_size": args.test_size,
            "random_state": args.seed,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
