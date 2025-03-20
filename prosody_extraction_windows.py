import os
import numpy as np
import librosa
import json
import opensmile
from tqdm import tqdm
from encoding.config import DATA_DIR

# Initialize OpenSMILE with eGeMAPS feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def process_audio_windows(audio_path, window_size=2.0, step_size=1.0):
    """Extract OpenSMILE features for sliding windows.
    
    Args:
        audio_path: Path to the audio file
        window_size: Size of sliding windows in seconds (default: 2.0)
        step_size: Step size for window sliding in seconds (default: 1.0)
    
    Returns:
        List of dictionaries containing window features
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)  # Keep original sample rate
    
    window_features = []
    window_samples = int(window_size * sr)
    step_samples = int(step_size * sr)
    
    print(f"\nProcessing audio file: {audio_path}")
    print(f"Audio duration: {len(y)/sr:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Window size: {window_size} seconds")
    print(f"Step size: {step_size} seconds")
    
    # Process sliding windows
    for start in range(0, len(y) - window_samples, step_samples):
        end = start + window_samples
        window_audio = y[start:end]
        start_time, end_time = start / sr, end / sr
        
        try:
            # Process window with OpenSMILE
            features = smile.process_signal(window_audio, sr)
            
            if not features.empty:
                # Convert features to regular Python types for JSON serialization
                feature_dict = {
                    'window_start_time': float(start_time),
                    'window_end_time': float(end_time),
                    'features': {col: float(val) for col, val in zip(features.columns, features.iloc[0].values)}
                }
                window_features.append(feature_dict)
                
                if len(window_features) == 1:  # Print feature names for the first window
                    print(f"\nExtracted {len(features.columns)} OpenSMILE features:")
                    print(f"Feature names: {list(features.columns)}")
                
                print(f"Processed window {start_time:.2f}-{end_time:.2f}s")
            
        except Exception as e:
            print(f"Warning: Could not process window {start_time:.2f}-{end_time:.2f}s: {str(e)}")
            continue
    
    print(f"\nProcessed {len(window_features)} windows successfully")
    return window_features

def main():
    """Main function to process all stories."""
    # Setup directories
    stimuli_dir = os.path.join(DATA_DIR, "ds003020/stimuli")
    output_dir = os.path.join(DATA_DIR, "features/prosody/opensmile")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(stimuli_dir) if f.endswith('.wav')]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    for audio_file in tqdm(audio_files):
        story_name = audio_file[:-4]  # Remove .wav extension
        audio_path = os.path.join(stimuli_dir, audio_file)
        
        # Process audio with 2-second windows and 1-second step size
        window_features = process_audio_windows(audio_path, window_size=2.0, step_size=1.0)
        
        # Save features
        output_path = os.path.join(output_dir, f"{story_name}_opensmile_windows.json")
        with open(output_path, 'w') as f:
            json.dump(window_features, f, indent=2)
        
        print(f"Saved features to {output_path}")

if __name__ == "__main__":
    main() 