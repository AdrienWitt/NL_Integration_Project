import os
import json
from collections import Counter
from sklearn.model_selection import train_test_split

def get_max_stories_participants(main_folder, min_participants_ratio=0.8, max_missing_stories=5):
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
    
    # Maximize stories where participants miss at most max_missing_stories
    selected_stories = []
    max_stories = 0
    for story_count in range(len(candidate_stories), 0, -1):
        for stories in [candidate_stories[i:i+story_count] for i in range(len(candidate_stories) - story_count + 1)]:
            valid_participants = sum(
                1 for p, p_stories in participant_stories.items()
                if len(set(stories) - p_stories) <= max_missing_stories
            )
            if valid_participants >= min_participants and len(stories) > max_stories:
                max_stories = len(stories)
                selected_stories = stories
    
    # Identify subjects missing stories
    missing_info = {
        p: sorted(list(set(selected_stories) - p_stories))
        for p, p_stories in participant_stories.items()
        if set(selected_stories) - p_stories
    }
    
    return sorted(selected_stories), participant_stories, missing_info

def main(main_folder, test_size=0.2, min_participants_ratio=0.8, max_missing_stories=2, output_json="stories_split.json"):
    # Get stories, participant data, and missing info
    selected_stories, participant_stories, missing_info = get_max_stories_participants(
        main_folder, min_participants_ratio, max_missing_stories
    )
    
    # Split into train and test
    train_stories, test_stories = train_test_split(
        selected_stories, test_size=test_size, random_state=42
    )
    
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
        "train_stories": sorted(train_stories),
        "test_stories": sorted(test_stories),
        "participants": {}
    }
    
    # Populate participant info
    for participant, stories in participant_stories.items():
        available_stories = sorted(stories.intersection(selected_stories))
        result["participants"][participant] = {
            "folder": os.path.join(main_folder, participant),
            "available_stories": available_stories,
            "train_stories": [s for s in train_stories if s in available_stories],
            "test_stories": [s for s in test_stories if s in available_stories]
        }
    
    # Save full result to JSON file
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
    
    # Print summary
    print(f"{full_participants} subjects share {len(selected_stories)} stories "
          f"while {missing_participants} subjects miss {missing_stories_total} compared to all others")
    if missing_info:
        print("Subjects missing stories:")
        for participant, missing_stories in missing_info.items():
            print(f"  {participant}: {missing_stories}")
    print(f"Selected stories: {selected_stories}")
    print(f"Train stories: {train_stories}")
    print(f"Test stories: {test_stories}")
    print(f"Full results saved to {output_json}")
    
    return result

if __name__ == "__main__":
    main_folder = r"E:\NL\ds003020\derivative\data\preprocessed_data"  # Replace with your folder path
    main(main_folder)