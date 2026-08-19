"""
Build the story lists the encoding stage reads.

Writes two files into `ENCODING_SPLIT_DIR`:

`all_stories.json`
    Which stories each subject heard, plus per-story subject counts. This is
    the default input to `encoding.run_encoding`, which trains each subject on
    everything they heard and tests on the repeated story.

`common_stories_<n>_for_<k>_subjects.json`
    The largest story set shared by exactly `k` subjects, for analyses that
    need every subject fit on identical stimuli. There is a real trade-off
    here — more subjects means fewer shared stories — so the script reports the
    whole curve and writes one file per subject count.

Usage
-----
    python -m prep.make_encoding_splits
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from config import ENCODING_SPLIT_DIR, HELD_OUT_STORY, ensure_dirs
from .make_story_splits import discover_subject_stories


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fmri-dir", default=None)
    p.add_argument("--out", default=str(ENCODING_SPLIT_DIR))
    p.add_argument("--held-out", default=HELD_OUT_STORY)
    p.add_argument("--max-subject-sets", type=int, default=8,
                   help="largest subject-count for which to write a common-story file")
    args = p.parse_args(argv)

    from config import FMRI_DIR
    fmri_dir = Path(args.fmri_dir or FMRI_DIR)
    subject_stories = discover_subject_stories(fmri_dir)

    story_to_subjects = defaultdict(list)
    for subject, stories in subject_stories.items():
        for story in stories:
            story_to_subjects[story].append(subject)

    ensure_dirs(args.out)
    out_dir = Path(args.out)

    all_stories = {
        "dataset_info": {
            "total_participants": len(subject_stories),
            "total_unique_stories": len(story_to_subjects),
            "participants": sorted(subject_stories),
            "held_out_story": args.held_out,
            "description": "Stories available per participant",
        },
        "participants": {s: sorted(v) for s, v in sorted(subject_stories.items())},
        "story_statistics": {
            story: {"participants": sorted(subs), "count": len(subs)}
            for story, subs in sorted(story_to_subjects.items())
        },
        "all_stories": sorted(story_to_subjects),
    }
    path = out_dir / "all_stories.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_stories, f, indent=2)
    print(f"Wrote {path}")
    print(f"  {len(subject_stories)} subjects, {len(story_to_subjects)} stories")

    # How many stories are shared as the subject group grows.
    print(f"\n{'Subjects':>9s} {'Shared stories':>15s}  Best group")
    print("  " + "-" * 58)
    subjects = sorted(subject_stories)
    for k in range(2, min(len(subjects), args.max_subject_sets) + 1):
        best_group, best_shared = None, set()
        for group in combinations(subjects, k):
            shared = set.intersection(*(subject_stories[s] for s in group))
            shared = {s for s in shared if not s.startswith(args.held_out)}
            if len(shared) > len(best_shared):
                best_group, best_shared = group, shared

        if not best_group:
            continue
        print(f"{k:>9d} {len(best_shared):>15d}  {', '.join(best_group)}")

        train = sorted(best_shared)
        test = [args.held_out] if all(
            args.held_out in subject_stories[s] for s in best_group
        ) else []
        payload = {
            "dataset_info": {
                "participants": list(best_group),
                "n_participants": k,
                "n_train_stories": len(train),
                "held_out_story": args.held_out,
            },
            "train": {"stories": train},
            "test": {"stories": test},
        }
        name = f"common_stories_{len(train)}_for_{k}_subjects.json"
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"\nCommon-story files written to {out_dir}")


if __name__ == "__main__":
    main()
