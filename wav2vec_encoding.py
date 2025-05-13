import os
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import h5py
import logging
from os.path import join
from encoding.ridge_utils.stimulus_utils import load_simulated_trfiles
from encoding.config import DATA_DIR
import json

# Constants
SAMPLING_RATE = 16000
WINDOW_SIZE = 2.0  # 2 seconds, matches TR
WINDOW_SAMPLES = int(WINDOW_SIZE * SAMPLING_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def load_model_and_processor():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model_path = "baobab/layers_frozen_6/final_model"
    model = Wav2Vec2Model.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor

def process_audio_window(audio_segment, processor, model):
    inputs = processor(
        audio_segment.squeeze(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=WINDOW_SAMPLES
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state.cpu().numpy()
    return np.mean(last_hidden_state, axis=1)

def get_wav2vec_vectors(stories, audio_dir, processor, model):
    """Get Wav2Vec embeddings for windows aligned to TR times."""
    # Load TR timings from respdict.json
    with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
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
            start_time = tr_time  # Window starts at TR time
            start_sample = int(start_time * SAMPLING_RATE)
            end_sample = start_sample + WINDOW_SAMPLES

            # Extract window
            if end_sample <= num_samples:
                window = waveform[:, start_sample:end_sample]
            else:
                # Pad if window extends beyond audio
                window = waveform[:, start_sample:]
                pad_length = WINDOW_SAMPLES - window.shape[1]
                if pad_length > 0:
                    window = torch.nn.functional.pad(window, (0, pad_length))
            
            # Compute embedding
            embedding = process_audio_window(window, processor, model)
            embeddings.append(embedding)

        vectors[story] = np.vstack(embeddings)
        logging.info(f"Processed {story} with {len(embeddings)} TR-aligned windows")

    return vectors

def main():
    audio_dir = "stimuli_16k"
    output_dir = "features/wav2vec"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, processor = load_model_and_processor()

    # Get stories
    textgrid_dir = join(DATA_DIR, "ds003020/derivative/TextGrids")
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]

    # Get Wav2Vec embeddings aligned to TRs
    vectors = get_wav2vec_vectors(stories, audio_dir, processor, model)

    # Save results
    for story in stories:
        output_file = os.path.join(output_dir, f"{story}_embeddings.hf5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('data', data=vectors[story])
        logging.info(f"Saved TR-aligned embeddings for {story} with shape {vectors[story].shape}")

if __name__ == "__main__":
    main()