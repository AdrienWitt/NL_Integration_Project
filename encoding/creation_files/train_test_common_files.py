import os
import json
from collections import defaultdict
from itertools import combinations


def create_multiple_train_test_splits(main_folder, output_dir="derivative", verbose=True):
    """
    Creates train/test split JSON files with:
    - Training stories: common clean stories (no '_') across participants
    - Test stories: exactly the repeated test story (wheretheressmoke without suffix)
      → excluded from training to prevent leakage
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    participant_folders = [
        f for f in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, f))
    ]
    num_participants = len(participant_folders)
    if verbose:
        print(f"Found {num_participants} participants")

    # ────────────────────────────────────────────────
    # Collect stories per participant
    # ────────────────────────────────────────────────
    participant_stories = {}
    all_stories_set = set()

    for folder in participant_folders:
        folder_path = os.path.join(main_folder, folder)
        stories = set()
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.hf5'):
                base_name = f[:-4]  # remove .hf5
                all_stories_set.add(base_name)
                # Training stories: exclude anything with '_' **and** anything starting with wheretheressmoke
                if '_' not in base_name and not base_name.lower().startswith("wheretheressmoke"):
                    stories.add(base_name)
        participant_stories[folder] = stories
        if verbose:
            print(f" {folder}: {len(stories)} clean training stories")

    # ────────────────────────────────────────────────
    # Define the exact test story name (the repeated one)
    # ────────────────────────────────────────────────
    # Option A: hard-code the clean name (recommended if consistent)
    test_story_name = "wheretheressmoke"  # the one with individual_repeats

    # Option B: dynamically find the one without suffix (if naming is consistent)
    # test_story_candidates = [s for s in all_stories_set if s.lower() == "wheretheressmoke"]
    # test_story_name = test_story_candidates[0] if test_story_candidates else None

    test_stories = [test_story_name] if test_story_name in all_stories_set else []

    if verbose:
        if test_stories:
            print(f"\nUsing test story: {test_stories[0]}")
        else:
            print("\nWarning: No 'wheretheressmoke' found in dataset")

    # ────────────────────────────────────────────────
    # Find participant groups with common training stories
    # ────────────────────────────────────────────────
    groups_with_stories = []

    for k in range(num_participants, 1, -1):
        for combo in combinations(participant_folders, k):
            common = set.intersection(*(participant_stories[p] for p in combo))
            if len(common) > 0:
                groups_with_stories.append((combo, common, len(common)))

    # Sort by number of common stories, then group size
    groups_with_stories.sort(key=lambda x: (x[2], len(x[0])), reverse=True)

    # Keep unique story sets
    unique_groups = []
    seen = set()
    for p, s, n in groups_with_stories:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            unique_groups.append((p, s, n))

    if verbose:
        print(f"\nFound {len(unique_groups)} unique participant groups with common stories")

    # ────────────────────────────────────────────────
    # Create JSON files
    # ────────────────────────────────────────────────
    summary = {
        "dataset_info": {
            "total_participants": num_participants,
            "total_unique_stories": len(all_stories_set),
            "analysis_date": "2026-02-06",
            "total_groups": len(unique_groups)
        },
        "groups": []
    }

    for idx, (participants, train_stories_set, num_train) in enumerate(unique_groups, 1):
        result = {
            "dataset_info": {
                "num_participants": len(participants),
                "participants": sorted(participants),
                "train_stories_count": num_train,
            },
            "train": {
                "stories": sorted(list(train_stories_set))
            },
            "test": {
                "stories": test_stories,  # always the same repeated test story
                "note": "repeated story (use individual_repeats if available)"
            }
        }

        filename = f"train_test_split_{num_train}_stories_{len(participants)}_subs.json"
        path = os.path.join(output_dir, filename)

        counter = 1
        base = filename
        while os.path.exists(path):
            filename = base.replace(".json", f"_{counter}.json")
            path = os.path.join(output_dir, filename)
            counter += 1

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)

        if verbose:
            print(f"{idx}. Created {filename}")
            print(f"   → {num_train} train stories")
            print(f"   → test: {test_stories}")

        summary["groups"].append({
            "file": filename,
            "train_stories": num_train,
            "participants": len(participants)
        })

    # Save summary
    with open(os.path.join(output_dir, "train_test_splits_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    if verbose:
        print(f"\nCreated {len(unique_groups)} split files")
        print(f"Test story is always: {test_stories}")
    
    return summary


if __name__ == "__main__":
    # Update these paths to match your system
    main_folder = r"E:\NL\clean_nl_preproc\ds003020\derivative\preprocessed_data"
    output_dir = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\derivative"

    print("Creating multiple train/test split JSON files")
    print("=" * 60)

    create_multiple_train_test_splits(
        main_folder=main_folder,
        output_dir=output_dir,
        verbose=True
    )