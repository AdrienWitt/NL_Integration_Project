import os
import json
import re

def list_stories_fmri(main_folder, preprocessed_folder):
    story_data = {}

    # Loop through participant folders
    for participant in sorted(os.listdir(main_folder)):
        if participant.startswith("sub-"):
            participant_path = os.path.join(main_folder, participant)
            
            # Find session folders
            for session in sorted(os.listdir(participant_path)):
                if session.startswith("ses-") and session != "ses-1":
                    session_path = os.path.join(participant_path, session, "func")
                    
                    if os.path.exists(session_path):
                        for file in sorted(os.listdir(session_path)):
                            if file.endswith(".nii.gz"):
                                match = re.search(r'task-([a-zA-Z0-9_]+)(?:_run-\d+)?_bold.nii.gz', file)
                                if match:
                                    story_name = match.group(1)
                                    
                                    if participant not in story_data:
                                        story_data[participant] = {}
                                    
                                    if session not in story_data[participant]:
                                        story_data[participant][session] = []
                                    
                                    story_data[participant][session].append(story_name)
                
    return story_data

def split_train_test(story_data, preprocessed_folder, common_test_stories=["wheretheressmoke", "buck"], train_ratio=0.8):
    train_test_split = {}
    
    # First, identify all unique stories across all participants
    all_unique_stories = set()
    for participant, sessions in story_data.items():
        for session_stories in sessions.values():
            all_unique_stories.update(session_stories)
    
    # Create global train/test split
    # First, assign common test stories to test set
    global_test = set(common_test_stories).intersection(all_unique_stories)
    
    # Get remaining stories
    remaining_stories = sorted(list(all_unique_stories - global_test))
    
    # Determine how many more stories should go to test set
    total_stories = len(all_unique_stories)
    target_test_size = int(total_stories * (1 - train_ratio))
    additional_test_needed = max(0, target_test_size - len(global_test))
    
    # Add more stories to global test set
    global_test.update(remaining_stories[:additional_test_needed])
    
    # Everything else goes to global train set
    global_train = set(all_unique_stories) - global_test
    
    print(f"Global train/test split: {len(global_train)} train stories, {len(global_test)} test stories")
    print(f"Test stories: {sorted(global_test)}")
    
    # Now apply this global split to each participant's data
    for participant, sessions in story_data.items():
        train_test_split[participant] = {}
        
        # Track which stories have been assigned for this participant
        participant_assigned_stories = set()
        
        # Process sessions in order
        for session in sorted(sessions.keys()):
            session_stories = set(sessions[session])
            
            # Filter to include only those with preprocessed data
            preprocessed_files = os.listdir(os.path.join(preprocessed_folder, participant[4:]))
            preprocessed_stories = {s for s in session_stories if f"{s}.hf5" in preprocessed_files}
            
            # For this session, only include stories not already assigned to another session
            available_stories = preprocessed_stories - participant_assigned_stories
            
            # Split according to global train/test sets
            session_train = list(global_train.intersection(available_stories))
            session_test = list(global_test.intersection(available_stories))
            
            # Update assigned stories
            participant_assigned_stories.update(available_stories)
            
            # Add this session's split to the result
            train_test_split[participant][session] = [session_train, session_test]
        
        # Print stats for this participant
        participant_train = set()
        participant_test = set()
        for session, (train_stories, test_stories) in train_test_split[participant].items():
            participant_train.update(train_stories)
            participant_test.update(test_stories)
        
        total_train = len(participant_train)
        total_test = len(participant_test)
        if total_train + total_test > 0:  # Avoid division by zero
            print(f"Participant {participant}: {total_train} train stories ({total_train/(total_train+total_test):.2f}), {total_test} test stories ({total_test/(total_train+total_test):.2f})")
        else:
            print(f"Participant {participant}: No stories with preprocessed data")
    
    return train_test_split


main_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020"  # Change this to the correct path
preprocessed_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\ds003020\derivative\preprocessed_data"  # Path to preprocessed data
output_file = "derivative/train_test_split.json"

# Generate story data
story_data = list_stories_fmri(main_folder, preprocessed_folder)

# Perform train-test split
train_test_data = split_train_test(story_data, preprocessed_folder)
    
# Save the result to a file
with open(output_file, "w") as f:
    json.dump(train_test_data, f, indent=4)
    
print(f"Train-test split saved to {output_file}")


def check_story_split_consistency(train_test_split):
    train_stories_per_participant = {}
    test_stories_per_participant = {}

    # Collect train and test stories for each participant
    for participant, sessions in train_test_split.items():
        train_stories_per_participant[participant] = set()
        test_stories_per_participant[participant] = set()

        for session, (train_stories, test_stories) in sessions.items():
            train_stories_per_participant[participant].update(train_stories)
            test_stories_per_participant[participant].update(test_stories)

    # Check for conflicts
    conflicts = []
    for participant_1 in train_stories_per_participant:
        for participant_2 in test_stories_per_participant:
            if participant_1 != participant_2:
                # Stories in train for participant_1 but test for participant_2
                conflict_train = train_stories_per_participant[participant_1] & test_stories_per_participant[participant_2]
                if conflict_train:
                    conflicts.append(f"Conflict: {participant_1}'s TRAIN stories appear in {participant_2}'s TEST set: {conflict_train}")
                
                # Stories in test for participant_1 but train for participant_2
                conflict_test = test_stories_per_participant[participant_1] & train_stories_per_participant[participant_2]
                if conflict_test:
                    conflicts.append(f"Conflict: {participant_1}'s TEST stories appear in {participant_2}'s TRAIN set: {conflict_test}")
                    print(f"Conflict: {participant_1}'s TEST stories appear in {participant_2}'s TRAIN set: {conflict_test}")

    return conflicts

# Run the check
conflicts = check_story_split_consistency(train_test_data)
