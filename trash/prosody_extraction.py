import os
import numpy as np
import librosa
import json
import parselmouth
import opensmile
from tqdm import tqdm
from encoding.ridge_utils.story_utils import get_story_grids
from encoding.config import DATA_DIR

# Initialize OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def extract_prosody(y, sr):
    """Extract prosody features from audio segment, including pitch std, intensity std, and spectral centroid."""
    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        
        # Pitch extraction
        pitch_values = sound.to_pitch().selected_array['frequency']
        pitch_valid = pitch_values[~np.isnan(pitch_values)]
        
        # Intensity extraction
        intensity = sound.to_intensity().values
        hnr = sound.to_harmonicity_cc().values
        
        speech_rate = estimate_speech_rate(y, sr)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Calculate means and standard deviations
        features = {
            'pitch_mean': np.nan if pitch_valid.size == 0 else float(np.mean(pitch_valid)),
            'pitch_std': np.nan if pitch_valid.size == 0 else float(np.std(pitch_valid)),
            'intensity_mean': np.nan if intensity.size == 0 else float(np.mean(intensity)),
            'intensity_std': np.nan if intensity.size == 0 else float(np.std(intensity)),
            'hnr_mean': np.nan if hnr.size == 0 else float(np.mean(hnr)),
            'hnr_std': np.nan if hnr.size == 0 else float(np.std(hnr)),
            'speech_rate': np.nan if speech_rate == 0 else speech_rate,
            'spectral_centroid_mean': np.nan if spectral_centroid.size == 0 else float(np.mean(spectral_centroid)),
            'spectral_centroid_std': np.nan if spectral_centroid.size == 0 else float(np.std(spectral_centroid))
        }
        
        return features
    except Exception as e:
        print(f"Error extracting prosody: {e}")
        return {
            'pitch_mean': np.nan, 'pitch_std': np.nan,
            'intensity_mean': np.nan, 'intensity_std': np.nan,
            'hnr_mean': np.nan, 'hnr_std': np.nan,
            'speech_rate': np.nan,
            'spectral_centroid_mean': np.nan, 'spectral_centroid_std': np.nan
        }

def estimate_speech_rate(y, sr, frame_length=512, hop_length=128):
    """
    Estimate the speech rate of a given audio segment (word or window) using onset strength.
    
    Parameters:
    - y: Audio signal (numpy array)
    - sr: Sample rate
    - is_word: If True, optimize for single word; if False, optimize for speech window
    - frame_length: Frame length for onset strength calculation (number of samples)
    - hop_length: Hop length for onset strength calculation (number of samples)
    
    Returns:
    - speech_rate: Speech rate (peaks per second)
    """
    # Calculate onset strength instead of RMS for better syllable onset detection
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length
    )
    
    # Normalize onset envelope for robust peak detection
    onset_norm = (onset_env - np.mean(onset_env)) / (np.std(onset_env) if np.std(onset_env) > 0 else 1.0)
    
    # Calculate hop time in seconds for wait parameter
    hop_time = hop_length / sr  # Time per hop in seconds
    
    # Set wait parameter based on typical syllable duration (250 ms)
    # For words, slightly shorter wait (e.g., 200 ms) might suffice due to fewer syllables
    syllable_duration = 0.20
    wait_steps = max(3, int(syllable_duration / hop_time))  # Ensure minimum of 3 steps
    
    # Detect peaks in onset strength (representing syllable onsets)
    peaks = librosa.util.peak_pick(
        onset_norm,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=3,
        delta=0.05,  # Threshold for peak prominence
        wait=wait_steps  # Minimum time between peaks
    )
    
    # Calculate duration in seconds
    duration = len(y) / sr
    
    # Calculate speech rate (peaks per second)
    speech_rate = len(peaks) / duration if duration > 0 else np.nan
    
    return speech_rate

def process_audio(audio_path, transcript, window_size=2.0, extract_windows=True):
    """Extract prosody features for each word and optionally for 2-second windows.
    
    Args:
        audio_path: Path to the audio file
        transcript: List of (start_time, end_time, word) tuples
        window_size: Size of sliding windows in seconds
        extract_windows: Whether to extract features for sliding windows
    """
    y, sr = librosa.load(audio_path, sr=None)  # `sr=None` keeps the original sample rate
    
    word_features = []
    window_features = []
    
    print(f"\nProcessing audio file: {audio_path}")
    print(f"Number of words in transcript: {len(transcript)}")
    
    # Process each word in transcript
    for start_time, end_time, word in transcript:
        print(f"\nProcessing word: '{word}' (start: {start_time}, end: {end_time})")
        try:
            start, end = round(float(start_time) * sr), round(float(end_time) * sr)
            if start < 0 or end > len(y):
                print(f"Warning: Invalid start or end: {start}, {end}, audio length: {len(y)}")
                continue
        except Exception as e:
            print(f"Error in timestamp conversion: {e}")
            continue

        word_audio = y[start:end]
        
        if len(word_audio) < sr * 0.05:
            print(f"Skipping word '{word}' - too short ({len(word_audio)/sr:.3f} seconds)")
            continue
        
        features = extract_prosody(word_audio, sr)
        features.update({'word': word, 'start_time': start_time, 'end_time': end_time})
        
        # Extract OpenSMILE features for this word
        try:
            # Save the word audio segment temporarily
            temp_path = os.path.join(os.path.dirname(audio_path), f"temp_{word}.wav")
            librosa.output.write_wav(temp_path, word_audio, sr)
            
            print(f"Processing OpenSMILE for word '{word}':")
            print(f"Audio duration: {len(word_audio)/sr:.3f} seconds")
            print(f"Sample rate: {sr} Hz")
            
            # Process the word segment with OpenSMILE
            word_opensmile = smile.process_file(temp_path)
            
            print(f"OpenSMILE output shape: {word_opensmile.shape if not word_opensmile.empty else 'Empty'}")
            
            # Clean up temporary file
            os.remove(temp_path)
            
            if not word_opensmile.empty:
                # Create a dictionary mapping feature names to their values, converting to native Python float
                feature_dict = {col: float(val) for col, val in zip(word_opensmile.columns, word_opensmile.iloc[0].values)}
                features['opensmile'] = feature_dict
                print(f"Successfully extracted {len(feature_dict)} OpenSMILE features")
            else:
                features['opensmile'] = None
                print(f"Warning: No OpenSMILE features available for word '{word}'")
        except Exception as e:
            print(f"Warning: Could not process OpenSMILE features for word '{word}': {str(e)}")
            features['opensmile'] = None
        
        # Print the extracted prosody features
        print(f"Word: {word}, Pitch Mean: {features['pitch_mean']:.2f}, "
              f"Pitch Std: {features['pitch_std']:.2f}, Intensity Mean: {features['intensity_mean']:.2f}, "
              f"Intensity Std: {features['intensity_std']:.2f}, HNR Mean: {features['hnr_mean']:.2f}, "
              f"HNR Std: {features['hnr_std']:.2f}, Speech Rate: {features['speech_rate']:.2f}, "
              f"Spectral Centroid Mean: {features['spectral_centroid_mean']:.2f}, "
              f"Spectral Centroid Std: {features['spectral_centroid_std']:.2f}")
        
        word_features.append(features)
    
    # Process 2-second sliding windows if requested
    if extract_windows:
        print("\nProcessing sliding windows...")
        window_samples = int(window_size * sr)
        step_samples = int(window_size * sr / 2)  # 50% overlap
        
        for start in range(0, len(y) - window_samples, step_samples):
            end = start + window_samples
            window_audio = y[start:end]
            start_time, end_time = start / sr, end / sr
            
            features = extract_prosody(window_audio, sr)
            features.update({'window_start_time': start_time, 'window_end_time': end_time})
            
            # Extract OpenSMILE features for this window
            try:
                # Save the window audio segment temporarily
                temp_path = os.path.join(os.path.dirname(audio_path), f"temp_window_{start_time:.2f}.wav")
                librosa.output.write_wav(temp_path, window_audio, sr)
                
                # Process the window segment with OpenSMILE
                window_opensmile = smile.process_file(temp_path)
                
                # Clean up temporary file
                os.remove(temp_path)
                
                if not window_opensmile.empty:
                    # Create a dictionary mapping feature names to their values, converting to native Python float
                    feature_dict = {col: float(val) for col, val in zip(window_opensmile.columns, window_opensmile.iloc[0].values)}
                    features['opensmile'] = feature_dict
                else:
                    features['opensmile'] = None
            except Exception as e:
                print(f"Warning: Could not process OpenSMILE features for window {start_time:.2f}-{end_time:.2f}: {str(e)}")
                features['opensmile'] = None
            
            # Print the extracted prosody features for the window
            print(f"Window {start_time:.2f}-{end_time:.2f}s, Pitch Mean: {features['pitch_mean']:.2f}, "
                  f"Pitch Std: {features['pitch_std']:.2f}, Intensity Mean: {features['intensity_mean']:.2f}, "
                  f"Intensity Std: {features['intensity_std']:.2f}, HNR Mean: {features['hnr_mean']:.2f}, "
                  f"HNR Std: {features['hnr_std']:.2f}, Speech Rate: {features['speech_rate']:.2f}, "
                  f"Spectral Centroid Mean: {features['spectral_centroid_mean']:.2f}, "
                  f"Spectral Centroid Std: {features['spectral_centroid_std']:.2f}")
            
            window_features.append(features)
    
    return word_features, window_features

def main():
    """Main function to process all stories."""
    textgrid_dir = os.path.join(DATA_DIR, "ds003020/derivative/TextGrids")
    stimuli_dir = os.path.join(DATA_DIR, "ds003020/stimuli")
    output_dir_word = os.path.join(DATA_DIR, "features/prosody/word_level")
    output_dir_window = os.path.join(DATA_DIR, "features/prosody/window_level")
    os.makedirs(output_dir_word, exist_ok=True)
    os.makedirs(output_dir_window, exist_ok=True)
    
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]

    transcripts = get_story_grids(stories, DATA_DIR)
    
    # Set this to False if you don't want window-level features
    extract_windows = False
    
    for story in tqdm(stories):
        audio_path = os.path.join(stimuli_dir, f"{story}.wav")
        word_features, window_features = process_audio(audio_path, transcripts[story], extract_windows=extract_windows)
        
        with open(os.path.join(output_dir_word, f"{story}_prosody.json"), 'w') as f:
            json.dump(word_features, f, indent=2)
        
        if extract_windows:
            with open(os.path.join(output_dir_window, f"{story}_window_prosody.json"), 'w') as f:
                json.dump(window_features, f, indent=2)

def extract_debug():
    """Run the script on one file for debugging."""
    sample_story = "sample_audio"
    textgrid_dir = os.path.join(DATA_DIR, "ds003020/derivative/TextGrids")
    stimuli_dir = os.path.join(DATA_DIR, "ds003020/stimuli")
    
    transcript = get_story_grids([sample_story], DATA_DIR)[sample_story]
    audio_path = os.path.join(stimuli_dir, f"{sample_story}.wav")
    process_audio(audio_path, transcript)
    
if __name__ == "__main__":
    main()