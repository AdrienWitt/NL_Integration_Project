import os
import numpy as np
import opensmile  # NEW!
import torchaudio
import h5py
import logging
from os.path import join
from encoding.ridge_utils.stimulus_utils import load_simulated_trfiles
from encoding.config import REPO_DIR
import json
import torch

# Constants
SAMPLING_RATE = 16000
WINDOW_SIZE = 2.0  # 2 seconds, matches TR
WINDOW_SAMPLES = int(WINDOW_SIZE * SAMPLING_RATE)

logging.basicConfig(level=logging.INFO)
logging.info(f"Using OpenSMILE eGeMAPSv02")

def load_opensmile():
    """Load OpenSMILE with eGeMAPSv02 functionals."""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    return smile

def process_audio_window(audio_segment, smile):
    """Extract OpenSMILE features for 2s audio window."""
    # Convert to numpy (OpenSMILE expects [samples])
    audio_np = audio_segment.squeeze().numpy()
    
    # Extract features
    features = smile.process_signal(audio_np, SAMPLING_RATE)
    
    # Return as [1, num_features] to match Wav2Vec shape convention
    return features.values.reshape(1, -1)

def get_opensmile_vectors(stories, audio_dir, smile):
    """Get OpenSMILE embeddings for windows aligned to TR times."""
    # Load TR timings from respdict.json
    with open(join(REPO_DIR, "ds003020/derivative/respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5)

    vectors = {}
    for story in stories:
        print(f"process {story}")
        audio_path = os.path.join(audio_dir, f"{story}.wav")
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != SAMPLING_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)
            waveform = resampler(waveform)

        num_samples = waveform.shape[1]
        tr_times = trfiles[story][0].get_reltriggertimes() + trfiles[story][0].soundstarttime
        num_trs = len(tr_times)
        embeddings = []

        # Process 2-second windows starting at each TR time
        for tr_time in tr_times:
            start_time = tr_time
            start_sample = int(start_time * SAMPLING_RATE)
            end_sample = start_sample + WINDOW_SAMPLES

            # Extract window
            if end_sample <= num_samples:
                window = waveform[:, start_sample:end_sample]
            else:
                window = waveform[:, start_sample:]
                pad_length = WINDOW_SAMPLES - window.shape[1]
                if pad_length > 0:
                    window = torch.nn.functional.pad(window, (0, pad_length))
            
            # Compute OpenSMILE features
            embedding = process_audio_window(window, smile)
            embeddings.append(embedding)

        vectors[story] = np.vstack(embeddings)
        logging.info(f"Processed {story} with {len(embeddings)} TR-aligned windows")

    return vectors

def main():
    audio_dir = "stimuli_16k"
    output_dir = "features/opensmile"  # CHANGED
    os.makedirs(output_dir, exist_ok=True)

    # Load OpenSMILE
    smile = load_opensmile()

    # Get stories
    textgrid_dir = join(REPO_DIR, "ds003020/derivative/TextGrids")
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]

    # Get OpenSMILE embeddings aligned to TRs
    vectors = get_opensmile_vectors(stories, audio_dir, smile)

    # Save results - EXACTLY like Wav2Vec
    for story in stories:
        output_file = os.path.join(output_dir, f"{story}_embeddings.hf5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('data', data=vectors[story])
        logging.info(f"Saved TR-aligned embeddings for {story} with shape {vectors[story].shape}")

if __name__ == "__main__":
    main()