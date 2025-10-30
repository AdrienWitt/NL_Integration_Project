import os
import glob
import json
import shutil
import logging

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def copy_story_files(stories, source_dir, dest_dir):
    """
    Copy .hf5 files for selected stories from source to destination folder, including duplicates (_1, _2, etc.).

    Parameters
    ----------
    stories : list
        List of story identifiers.
    source_dir : str
        Source directory containing .hf5 files.
    dest_dir : str
        Destination directory for selected .hf5 files.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    # Track copied and missing/failed files
    copied_files = []
    missing_items = []  # Will include stories (if no files found) and filenames (if copy fails)
    
    # Copy .hf5 files for each story, including duplicates (_1, _2, etc.)
    for story in stories:
        # Search for files matching the story name with wildcard for duplicates (e.g., story.hf5, story_1.hf5)
        pattern = os.path.join(source_dir, f"{story}*.hf5")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            logging.warning(f"No .hf5 files found for story {story} in {source_dir}")
            missing_items.append(story)
            continue
        
        for src_file in matching_files:
            filename = os.path.basename(src_file)
            dest_file = os.path.join(dest_dir, filename)
            
            try:
                shutil.copy2(src_file, dest_file)  # Use copy2 to preserve metadata; overwrites if exists
                logging.info(f"Copied {src_file} to {dest_file}")
                copied_files.append(filename)
            except Exception as e:
                logging.error(f"Failed to copy {src_file} to {dest_file}: {e}")
                missing_items.append(filename)
    
    logging.info(f"Copied {len(copied_files)} files to {dest_dir}: {copied_files}")
    if missing_items:
        logging.warning(f"Issues with {len(missing_items)} items for this subject (stories or files): {missing_items}")
    
    return copied_files, missing_items

if __name__ == "__main__":
    setup_logging()
    
    # Define base paths
    bids_dir = "E:/NL/ds003020"
    source_base_dir = os.path.join(bids_dir, "derivative", "preprocessed_data")
    dest_base_dir = os.path.join(bids_dir, "derivative", "to_send")
    json_path = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\derivative\common_stories_27.json"
    
    # Load the JSON data once (contains all subjects)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        participants = data.get("participants", {})
        if not participants:
            raise ValueError(f"No participants found in {json_path}")
        logging.info(f"Loaded data for {len(participants)} subjects from {json_path}")
    except Exception as e:
        logging.error(f"Failed to load JSON {json_path}: {e}")
        raise
    
    # Track overall results
    total_copied_files = []
    total_missing_items = []
    
    # Process all subjects
    for subject_key in participants:
        stories = participants[subject_key].get("stories", [])
        if not stories:
            logging.warning(f"No stories found for subject {subject_key}")
            continue
        
        source_dir = os.path.join(source_base_dir, subject_key)
        dest_dir = os.path.join(dest_base_dir, subject_key)
        
        logging.info(f"Processing subject {subject_key} with {len(stories)} stories: {stories}")
        
        copied_files, missing_items = copy_story_files(stories, source_dir, dest_dir)
        
        total_copied_files.extend(copied_files)
        total_missing_items.extend(missing_items)
    
    # Overall summary
    logging.info(f"Overall process complete: Copied {len(total_copied_files)} files across all subjects")
    if total_missing_items:
        logging.warning(f"Overall issues with {len(total_missing_items)} items across all subjects")