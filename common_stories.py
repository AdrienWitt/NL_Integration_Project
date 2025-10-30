import os
import json
from collections import Counter

def get_common_stories(main_folder, min_participants_ratio=0.8, max_missing_stories=5, output_json="common_stories.json", target_stories=None):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get participant subfolders
    participant_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    num_participants = len(participant_folders)
    min_participants = int(num_participants * min_participants_ratio)
    
    # Collect stories per participant, excluding repeats
    participant_stories = {}
    all_stories = Counter()
    for folder in participant_folders:
        folder_path = os.path.join(main_folder, folder)
        stories = {f.split('.')[0] for f in os.listdir(folder_path) if '_' not in f}
        participant_stories[folder] = stories
        all_stories.update(stories)
    
    # Find stories shared by at least min_participants
    candidate_stories = [story for story, count in all_stories.items() if count >= min_participants]
    
    # Maximize stories or target specific number of stories
    selected_stories = []
    max_stories = 0
    max_valid_participants = 0
    for story_count in range(len(candidate_stories), 0, -1):
        if target_stories is not None and story_count != target_stories:
            continue  # Skip if not matching target story count
        for stories in [candidate_stories[i:i+story_count] for i in range(len(candidate_stories) - story_count + 1)]:
            valid_participants = sum(
                1 for p, p_stories in participant_stories.items()
                if len(set(stories) - p_stories) <= max_missing_stories
            )
            # Prioritize more participants if story count is equal or target is met
            if valid_participants >= min_participants and (
                (len(stories) > max_stories) or 
                (len(stories) == max_stories and valid_participants > max_valid_participants)
            ):
                max_stories = len(stories)
                max_valid_participants = valid_participants
                selected_stories = stories
    
    # Identify subjects missing stories
    missing_info = {
        p: sorted(list(set(selected_stories) - p_stories))
        for p, p_stories in participant_stories.items()
        if set(selected_stories) - p_stories
    }
    
    # Count participants with all stories and those missing some
    full_participants = sum(
        1 for p, stories in participant_stories.items()
        if set(selected_stories).issubset(stories)
    )
    missing_participants = len(missing_info)
    missing_stories_total = sum(len(stories) for stories in missing_info.values())
    
    # Create result dictionary
    result = {
        "selected_stories": sorted(selected_stories),
        "participants": {}
    }
    
    # Populate participant info
    for participant, stories in participant_stories.items():
        stories_shared = sorted(stories.intersection(selected_stories))
        result["participants"][participant] = {
            "folder": os.path.join(main_folder, participant),
            "stories": stories_shared  # Changed from available_stories to stories
        }
    
    # Save result to JSON file
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
    
    # Print summary
    print(f"\nOutput for {output_json}:")
    print(f"{full_participants} subjects share {len(selected_stories)} stories "
          f"while {missing_participants} subjects miss {missing_stories_total} compared to all others")
    print(f"Total participants included: {max_valid_participants}")
    if missing_info:
        print("Subjects missing stories:")
        for participant, missing_stories in missing_info.items():
            print(f"  {participant}: {missing_stories}")
    print(f"Selected stories: {selected_stories}")
    print(f"Results saved to {output_json}")
    
    return result

if __name__ == "__main__":
    main_folder = r"E:\NL\ds003020\derivative\data\preprocessed_data"
    derivative_folder = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\derivative"
    
    # Configuration for 25 stories (targeting 9 participants)
    get_common_stories(
        main_folder,
        min_participants_ratio=0.75,  # Lowered to include more participants
        max_missing_stories=5,
        output_json=os.path.join(derivative_folder, "common_stories_25.json"),
        target_stories=25  # Force exactly 25 stories
    )
    
    # Configuration for 27 stories (targeting 7 participants)
    get_common_stories(
        main_folder,
        min_participants_ratio=0.9,   # Higher to prioritize more stories, fewer participants
        max_missing_stories=2,
        output_json=os.path.join(derivative_folder, "common_stories_27.json"),
        target_stories=27  # Force exactly 27 stories
    )