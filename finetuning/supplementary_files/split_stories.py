import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────────────────────────
PREPROC_DIR  = r"E:\NL\clean_nl_preproc\ds003020\derivative\preprocessed_data"
MAIN_DIR     = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\finetuning"
OUTPUT_DIR   = "splits"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Stories to hold out as final test set ────────────────────────────────────
HELD_OUT_TEST = {"wheretheressmoke"}  # add more if needed

# ── Step 1: discover stories ──────────────────────────────────────────────────
subject_stories = defaultdict(set)

for subject in sorted(os.listdir(PREPROC_DIR)):
    subject_path = os.path.join(PREPROC_DIR, subject)
    if not os.path.isdir(subject_path) or not subject.startswith("UTS"):
        continue
    for fname in sorted(os.listdir(subject_path)):
        if fname.endswith(".hf5"):
            subject_stories[subject].add(fname.replace(".hf5", ""))

all_subjects = set(subject_stories.keys())
print(f"Found {len(all_subjects)} subjects: {sorted(all_subjects)}")
for subj in sorted(all_subjects):
    print(f"  {subj}: {len(subject_stories[subj])} stories")

# ── Step 2: universal stories ─────────────────────────────────────────────────
universal_stories = sorted(set.intersection(*subject_stories.values()))
print(f"\nUniversal stories ({len(universal_stories)}): {universal_stories}")

# Confirm held-out stories exist
for s in HELD_OUT_TEST:
    if s not in set(universal_stories):
        print(f"  WARNING: '{s}' is not a universal story — "
              f"not all subjects will have brain data for it")
    else:
        print(f"  ✓ '{s}' is universal — all subjects have brain data")

# ── Step 3: remove held-out from the pool entirely ───────────────────────────
all_heard             = sorted(set.union(*subject_stories.values()))
available_stories     = [s for s in all_heard if s not in HELD_OUT_TEST]
universal_available   = [s for s in universal_stories if s not in HELD_OUT_TEST]
non_universal_stories = [s for s in available_stories if s not in set(universal_available)]

print(f"\nHeld-out test stories  : {sorted(HELD_OUT_TEST)}")
print(f"Universal (available)  : {len(universal_available)}")
print(f"Non-universal          : {len(non_universal_stories)}")

# ── Step 4: split non-universal into train + val ──────────────────────────────
extra_train_stories, val_stories = train_test_split(
    non_universal_stories,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_stories = sorted(set(universal_available) | set(extra_train_stories))
val_stories   = sorted(val_stories)
test_stories  = sorted(HELD_OUT_TEST)

# ── Step 5: sanity checks ─────────────────────────────────────────────────────
assert not set(train_stories) & set(val_stories),  "ERROR: train/val overlap!"
assert not set(train_stories) & set(test_stories), "ERROR: train/test overlap!"
assert not set(val_stories)   & set(test_stories), "ERROR: val/test overlap!"
assert all(s in train_stories for s in universal_available), \
    "ERROR: some universal stories not in train!"
print("\n✓ No train/val/test overlap")
print("✓ All available universal stories are in training")

print(f"\n{'='*60}")
print(f"Final train : {len(train_stories)} stories")
print(f"Final val   : {len(val_stories)} stories")
print(f"Final test  : {len(test_stories)} stories → {test_stories}")
print(f"{'='*60}")

# Per-subject coverage
print(f"\nPer-subject story coverage:")
print(f"  {'Subject':10s} {'Total':>6s} {'Train':>7s} {'Val':>6s} {'Test':>6s}")
print(f"  {'-'*40}")
for subject in sorted(all_subjects):
    s = subject_stories[subject]
    print(f"  {subject:10s} {len(s):>6d} "
          f"{len(s & set(train_stories)):>7d} "
          f"{len(s & set(val_stories)):>6d} "
          f"{len(s & set(test_stories)):>6d}")

# ── Step 6: save ──────────────────────────────────────────────────────────────
split_dir = os.path.join(MAIN_DIR, OUTPUT_DIR)
os.makedirs(split_dir, exist_ok=True)

split = {
    "train":             train_stories,
    "val":               val_stories,
    "test":              test_stories,
    "n_train":           len(train_stories),
    "n_val":             len(val_stories),
    "n_test":            len(test_stories),
    "universal_stories": universal_available,
    "held_out_test":     sorted(HELD_OUT_TEST),
    "test_size":         TEST_SIZE,
    "random_state":      RANDOM_STATE,
}

split_path = os.path.join(split_dir, "stories_split.json")
with open(split_path, "w") as f:
    json.dump(split, f, indent=2)

print(f"\nSplit saved to: {split_path}")