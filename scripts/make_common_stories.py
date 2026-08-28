"""
Derive the true common-story list from `all_stories.json`.

Group-level results need every subject fitted on the same stories: subjects
heard between 26 and 84, and a model trained on 83 stories predicts better than
one trained on 25 for reasons that have nothing to do with the features. Any
average over subjects, or any group voxel map, has to hold that constant.

The shipped `common_stories_25.json` cannot do that job. Its participant keys
are `sub-UTS01` where the rest of the codebase uses `UTS01`, so
`stories_for_subject` raises KeyError on every subject — and its story list is
not the intersection anyway: it contains `life` (UTS04 never heard it) and
`fromboyhoodtofatherhood` (UTS09 never heard it), while omitting `legacy` and
`thatthingonmyarm`, which all nine did hear. Two subjects therefore came out
with 24 stories and the rest with 25, which is the opposite of the point.

This computes the intersection instead, so it cannot drift from the source:

    25 stories shared by all 9 subjects, `wheretheressmoke` among them
    -> 24 training stories once the held-out story is removed

Writing it in the same shape `stories_for_subject` expects means every subject
gets an identical list, and the file can be regenerated rather than trusted.

    python scripts/make_common_stories.py
    python -m encoding.run_encoding --stories-json common_stories_all9.json ...
"""

import argparse
import functools
import json
from pathlib import Path

from config import ENCODING_SPLIT_DIR, HELD_OUT_STORY


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="all_stories.json")
    p.add_argument("--out", default="common_stories_all9.json")
    p.add_argument("--subjects", default=None,
                   help="comma-separated subset; default is every participant "
                        "in the source file")
    args = p.parse_args()

    src = Path(ENCODING_SPLIT_DIR) / args.source
    data = json.loads(src.read_text(encoding="utf-8"))
    part = data["participants"]

    wanted = ([s.strip() for s in args.subjects.split(",")]
              if args.subjects else sorted(part))
    per = {}
    for s in wanted:
        entry = part[s]
        per[s] = set(entry["stories"] if isinstance(entry, dict) else entry)

    common = sorted(functools.reduce(lambda a, b: a & b, per.values()))
    print(f"source   : {src}")
    for s in wanted:
        print(f"  {s}: {len(per[s]):>3} stories")
    print(f"\ncommon to all {len(wanted)}: {len(common)}")
    print(f"  {HELD_OUT_STORY} present: {HELD_OUT_STORY in common}")
    n_train = len(common) - (1 if HELD_OUT_STORY in common else 0)
    print(f"  training stories after holding it out: {n_train}")

    if HELD_OUT_STORY not in common:
        # Every reported number is scored on this story; a common set without it
        # would silently fall back to a different test set per subject.
        raise SystemExit(
            f"{HELD_OUT_STORY} is not shared by all subjects — the held-out "
            f"story must be in the common set or the evaluation is not common."
        )

    out = Path(ENCODING_SPLIT_DIR) / args.out
    out.write_text(json.dumps({
        "derived_from": args.source,
        "n_common": len(common),
        "held_out_story": HELD_OUT_STORY,
        "n_train_after_holdout": n_train,
        "common_stories": common,
        "participants": {s: {"stories": common} for s in wanted},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
