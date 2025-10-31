import os
import json
from collections import Counter, defaultdict
from itertools import combinations

def find_optimal_common_stories(main_folder, output_dir="derivative", verbose=True):
    """
    Find the maximum number of common stories for each possible participant count.
    Creates separate JSON files for each optimal configuration.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get participant subfolders
    participant_folders = [f for f in os.listdir(main_folder) 
                          if os.path.isdir(os.path.join(main_folder, f))]
    num_participants = len(participant_folders)
    
    if verbose:
        print(f"Found {num_participants} participants")
    
    # Collect stories per participant (excluding repeats with '_')
    participant_stories = {}
    for folder in participant_folders:
        folder_path = os.path.join(main_folder, folder)
        stories = {f.split('.')[0] for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) and '_' not in f}
        participant_stories[folder] = stories
        if verbose:
            print(f"  {folder}: {len(stories)} stories")
    
    # Build a mapping: story -> set of participants who have it
    story_participants = defaultdict(set)
    for participant, stories in participant_stories.items():
        for story in stories:
            story_participants[story].add(participant)
    
    if verbose:
        print(f"\nTotal unique stories: {len(story_participants)}")
    
    # For each possible participant count, find maximum common stories
    results_summary = []
    
    for target_count in range(num_participants, 1, -1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Finding optimal stories for {target_count} participants...")
        
        max_stories = 0
        best_participants = None
        best_common_stories = None
        
        # Try all combinations of participants
        for participant_combo in combinations(participant_folders, target_count):
            # Find stories common to all participants in this combination
            common = set.intersection(*[participant_stories[p] for p in participant_combo])
            
            if len(common) > max_stories:
                max_stories = len(common)
                best_participants = participant_combo
                best_common_stories = common
        
        if max_stories == 0:
            if verbose:
                print(f"  No common stories found for {target_count} participants")
            continue
        
        # Create result dictionary
        result = {
            "configuration": {
                "num_participants": target_count,
                "num_stories": max_stories,
                "total_participants_in_dataset": num_participants
            },
            "selected_stories": sorted(best_common_stories),
            "participants": {}
        }
        
        # Add participant information
        for participant in best_participants:
            result["participants"][participant] = {
                "folder": os.path.join(main_folder, participant),
                "stories": sorted(best_common_stories),
                "total_stories_available": len(participant_stories[participant])
            }
        
        # Save to JSON
        output_file = os.path.join(output_dir, f"common_stories_{max_stories}_for_{target_count}_participants.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        
        # Store summary
        results_summary.append({
            "participants": target_count,
            "stories": max_stories,
            "participant_list": sorted(best_participants),
            "output_file": output_file
        })
        
        if verbose:
            print(f"  ✓ Found {max_stories} common stories for {target_count} participants")
            print(f"    Participants: {', '.join(sorted(best_participants))}")
            print(f"    Saved to: {output_file}")
    
    # Create summary JSON
    summary_file = os.path.join(output_dir, "common_stories_summary.json")
    summary = {
        "total_participants": num_participants,
        "configurations": results_summary,
        "analysis_date": "2025-10-31"
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print final summary table
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY OF OPTIMAL CONFIGURATIONS")
        print(f"{'='*60}")
        print(f"{'Participants':<15} {'Stories':<10} {'Output File'}")
        print(f"{'-'*60}")
        for item in results_summary:
            filename = os.path.basename(item['output_file'])
            print(f"{item['participants']:<15} {item['stories']:<10} {filename}")
        print(f"\nSummary saved to: {summary_file}")
    
    return results_summary


def analyze_story_overlap(main_folder, output_file="story_overlap_analysis.json"):
    """
    Additional analysis: Show how many participants have each story.
    Useful for understanding the data distribution.
    """
    participant_folders = [f for f in os.listdir(main_folder) 
                          if os.path.isdir(os.path.join(main_folder, f))]
    
    # Collect stories per participant
    participant_stories = {}
    story_counts = Counter()
    
    for folder in participant_folders:
        folder_path = os.path.join(main_folder, folder)
        stories = {f.split('.')[0] for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) and '_' not in f}
        participant_stories[folder] = stories
        story_counts.update(stories)
    
    # Organize stories by participant count
    stories_by_count = defaultdict(list)
    for story, count in story_counts.items():
        stories_by_count[count].append(story)
    
    analysis = {
        "total_participants": len(participant_folders),
        "distribution": {}
    }
    
    for count in sorted(stories_by_count.keys(), reverse=True):
        stories = sorted(stories_by_count[count])
        analysis["distribution"][f"{count}_participants"] = {
            "count": len(stories),
            "stories": stories
        }
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=4)
    
    print(f"\nStory overlap analysis saved to: {output_file}")
    print("\nDistribution of stories by participant count:")
    for count in sorted(stories_by_count.keys(), reverse=True):
        print(f"  {count} participants: {len(stories_by_count[count])} stories")
    
    return analysis


if __name__ == "__main__":
    # Update these paths to match your system
    main_folder = r"E:\NL\ds003020\derivative\data\preprocessed_data"
    output_dir = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\derivative"
    
    # Find optimal configurations for all participant counts
    print("FINDING OPTIMAL COMMON STORIES FOR ALL PARTICIPANT COUNTS")
    print("="*60)
    results = find_optimal_common_stories(main_folder, output_dir, verbose=True)
    
    # Optional: Run additional analysis
    print("\n" + "="*60)
    print("ANALYZING STORY OVERLAP DISTRIBUTION")
    print("="*60)
    analyze_story_overlap(main_folder, os.path.join(output_dir, "story_overlap_analysis.json"))