import os
import json
import logging
import numpy as np
import torch
import torchaudio
import librosa
import pandas as pd
import parselmouth
from scipy.stats import pearsonr, skew, kurtosis

from transformers import Wav2Vec2Model, Wav2Vec2Processor

from transformers import HubertModel, HubertProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# # Load wav2vec2 model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

def extract_prosody_features(y, sr):
    """Extract prosody features from an audio segment."""
    # Ensure the waveform is in float64 format for Parselmouth
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

    # Pitch (F0)
    pitch_values = snd.to_pitch().selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # Remove zero values

    # Check if pitch_values is empty before computing statistical features
    if pitch_values.size == 0:
        # If no valid pitch values are found, set them to NaN or some default value
        pitch_mean = pitch_std = pitch_skew = pitch_kurt = pitch_min = pitch_max = np.nan
    else:
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        pitch_skew = skew(pitch_values)
        pitch_kurt = kurtosis(pitch_values)
        pitch_min = np.min(pitch_values)
        pitch_max = np.max(pitch_values)

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    # Speech Rate (Tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]

    # Spectral Centroid (Energy distribution)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_std = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Zero-Crossing Rate
    zero_crossings = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate = np.mean(zero_crossings)

    # Harmonics-to-Noise Ratio (HNR)
    hnr_values = librosa.effects.harmonic(y)
    hnr = np.mean(hnr_values)

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_skew": pitch_skew,
        "pitch_kurt": pitch_kurt,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "energy": energy,
        "speech_rate": tempo,
        "spectral_centroid": spectral_centroid,
        "spectral_centroid_std": spectral_centroid_std,
        "zero_crossing_rate": zero_crossing_rate,
        "hnr": hnr
    }

def get_wav2vec_embeddings(y, sr):
    """Extract wav2vec2 embeddings from all layers for an audio segment."""
    waveform = torch.tensor(y).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Return embeddings maintaining time dimension
    return [layer.squeeze().mean(dim=0).cpu().numpy() for layer in outputs.hidden_states]

def compute_correlations(all_embeddings, all_prosodic):
    """Compute correlations between embeddings and prosody features, handling NaNs properly."""

    feature_names = list(all_prosodic[0].keys())
    prosody_array = np.array([[feat[name] for name in feature_names] for feat in all_prosodic])

    # Remove NaN-containing rows from both prosody features and embeddings
    valid_rows = ~np.isnan(prosody_array).any(axis=1)
    prosody_array = prosody_array[valid_rows]
    all_embeddings = [all_embeddings[i] for i in range(len(all_embeddings)) if valid_rows[i]]

    correlations = {}

    # Process each layer separately
    num_layers = len(all_embeddings[0])  # Number of layers in the first segment
    for layer_idx in range(num_layers):
        # Collect embeddings for this layer across all segments
        layer_embeddings = [segment[layer_idx] for segment in all_embeddings]
        
        # Average across all dimensions for each segment
        layer_means = np.mean(layer_embeddings, axis=1)
        
        layer_correlations = {}
        for j, feature_name in enumerate(feature_names):
            feature_values = prosody_array[:, j]
            
            # Compute correlation if both arrays have variation
            if np.std(feature_values) > 0 and np.std(layer_means) > 0:
                try:
                    corr, p_value = pearsonr(feature_values, layer_means)
                except Exception as e:
                    print(f"Error computing correlation for layer {layer_idx}: {e}")
                    corr, p_value = 0, 1.0
            else:
                corr, p_value = 0, 1.0
                
            layer_correlations[feature_name] = {
                'correlation': float(corr),
                'p_value': float(p_value)
            }
        
        correlations[f'Layer_{layer_idx}'] = layer_correlations

    return correlations

def analyze_audio(audio_path, return_data=False):
    """Analyze an audio file in fixed 2-second segments."""
    y, sr = librosa.load(audio_path, sr=None)  # Load audio here
    total_duration = librosa.get_duration(y=y, sr=sr)
    segment_duration = 2.0
    all_prosodic = []
    all_embeddings = []
    
    for start in np.arange(0, total_duration, segment_duration):
        end = min(start + segment_duration, total_duration)
        start_sample, end_sample = int(start * sr), int(end * sr)
        segment = y[start_sample:end_sample]
        prosody_features = extract_prosody_features(segment, sr)
        embeddings = get_wav2vec_embeddings(segment, sr)
        all_prosodic.append(prosody_features)
        all_embeddings.append(embeddings)

    if return_data:
        return all_embeddings, all_prosodic
    else:
        return compute_correlations(all_embeddings, all_prosodic) if all_prosodic else {}

def main(data_dir, model):
    output_dir = os.path.join(data_dir, "features", f"{model}_prossody_correlations")
    os.makedirs(output_dir, exist_ok=True)
    textgrid_dir = os.path.join(data_dir, "ds003020/derivative/TextGrids")
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]
    all_results = {}
    
    # Initialize lists to store all embeddings and features
    all_embeddings = []
    all_prosodic = []
    
    for story in stories:
        logging.info(f"Analyzing {story}...")
        audio_path = os.path.join(data_dir, "ds003020/stimuli", f"{story}.wav")
        correlations = analyze_audio(audio_path)
        all_results[story] = correlations
        json_path = os.path.join(output_dir, f"{story}_prosody_correlations.json")
        with open(json_path, 'w') as f:
            json.dump(correlations, f, indent=2)
        logging.info(f"Saved correlations for {story}")
        
        # Collect embeddings and features for global correlation
        story_embeddings, story_prosodic = analyze_audio(audio_path, return_data=True)
        all_embeddings.extend(story_embeddings)
        all_prosodic.extend(story_prosodic)
    
    # Compute global correlations
    global_correlations = compute_correlations(all_embeddings, all_prosodic)
    global_json_path = os.path.join(output_dir, "global_prosody_correlations.json")
    with open(global_json_path, 'w') as f:
        json.dump(global_correlations, f, indent=2)
    logging.info("Saved global correlations.")
    
    summary_data = [{
        'story': story, 'layer': layer, 'feature': feature, 'correlation': val['correlation']
    } for story, result in all_results.items() for layer, feats in result.items() for feature, val in feats.items()]
    summary_data.extend({'story': 'global', 'layer': layer, 'feature': feature, 'correlation': val['correlation']} for layer, feats in global_correlations.items() for feature, val in feats.items())
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "prosody_correlations_summary.csv"), index=False)
    logging.info("Summary saved.")

if __name__ == "__main__":
    from encoding.config import DATA_DIR as data_dir
    main(data_dir, model)
