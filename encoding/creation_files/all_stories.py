import os
import json
from collections import defaultdict


def create_participant_stories_dict(main_folder, output_dir="derivative", verbose=True):
    """
    Creates a single JSON file with all stories available per participant.
    
    Output structure:
    {
        "dataset_info": {...},
        "participants": {
            "sub-01": ["story1", "story2", ...],
            "sub-02": ["story1", "story3", ...],
            ...
        },
        "story_statistics": {
            "story1": ["sub-01", "sub-02", ...],
            ...
        }
    }
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
        print("=" * 60)

    # ────────────────────────────────────────────────
    # Collect all stories per participant
    # ────────────────────────────────────────────────
    participant_stories = {}
    all_stories_set = set()
    story_to_participants = defaultdict(list)

    for folder in sorted(participant_folders):
        folder_path = os.path.join(main_folder, folder)
        stories = []
        
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.hf5'):
                base_name = f[:-4]  # remove .hf5
                stories.append(base_name)
                all_stories_set.add(base_name)
                story_to_participants[base_name].append(folder)
        
        participant_stories[folder] = sorted(stories)
        
        if verbose:
            print(f"{folder}: {len(stories)} stories")

    # ────────────────────────────────────────────────
    # Create comprehensive JSON output
    # ────────────────────────────────────────────────
    output_data = {
        "dataset_info": {
            "total_participants": num_participants,
            "total_unique_stories": len(all_stories_set),
            "analysis_date": "2026-02-06",
            "description": "All stories available for each participant"
        },
        "participants": participant_stories,
        "story_statistics": {
            story: {
                "participants": sorted(participants),
                "count": len(participants)
            }
            for story, participants in sorted(story_to_participants.items())
        },
        "all_stories": sorted(list(all_stories_set))
    }

    # ────────────────────────────────────────────────
    # Save to JSON file
    # ────────────────────────────────────────────────
    output_filename = "participant_stories_dict.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    if verbose:
        print("=" * 60)
        print(f"\nCreated: {output_filename}")
        print(f"Total unique stories: {len(all_stories_set)}")
        print(f"\nStory distribution:")
        print("-" * 60)
        
        # Show which stories appear for how many participants
        story_counts = defaultdict(int)
        for story, data in output_data["story_statistics"].items():
            count = data["count"]
            story_counts[count] += 1
        
        for count in sorted(story_counts.keys(), reverse=True):
            num_stories = story_counts[count]
            print(f"  {num_stories} stories appear in {count} participant(s)")
        
        print("\nAll unique stories:")
        print("-" * 60)
        for story in sorted(all_stories_set):
            count = len(story_to_participants[story])
            print(f"  {story}: {count} participants")

    return output_data


if __name__ == "__main__":
    # Update these paths to match your system
    main_folder = r"E:\NL\clean_nl_preproc\ds003020\derivative\preprocessed_data"
    output_dir = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\derivative"

    print("Creating participant stories dictionary")
    print("=" * 60)

    result = create_participant_stories_dict(
        main_folder=main_folder,
        output_dir=output_dir,
        verbose=True
    )