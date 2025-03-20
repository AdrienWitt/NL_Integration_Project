import os
import sys
import numpy as np
import h5py
import argparse
import json
import pathlib
from os.path import join, dirname
import logging
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from encoding.ridge_utils.DataSequence import DataSequence
from encoding.ridge_utils.textgrid import TextGrid
from encoding.ridge_utils.interpdata import lanczosinterp2D
from encoding.ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from encoding.ridge_utils.dsutils import make_word_ds
from encoding.config import DATA_DIR
from encoding.ridge_utils.story_utils import get_story_wordseqs

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Initialize wav2vec model and processor globally
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

def get_wav2vec_embeddings(audio_path, start_time, end_time):
    """Get wav2vec embeddings for a given audio segment.
    
    Args:
        audio_path: Path to the audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Embeddings for the audio segment
    """
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Convert times to samples
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Extract segment
    segment = waveform[:, start_sample:end_sample]
    
    # Process audio
    inputs = processor(segment, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last hidden state embeddings
    embeddings = outputs.last_hidden_state.cpu().numpy()
    
    # Average across time steps
    embeddings = np.mean(embeddings, axis=1)
    
    return embeddings

def get_wav2vec_vectors(stories, data_dir):
    """Get wav2vec embeddings for specified stories.
    
    Args:
        stories: List of stories to obtain vectors for
        data_dir: Directory containing the data
        
    Returns:
        Dictionary of {story: <float32>[num_segments, vector_size]}
    """
    wordseqs = get_story_wordseqs(stories, data_dir)
    vectors = {}
    
    for story in stories:
        # Get words and their timings
        words = wordseqs[story].data
        times = wordseqs[story].data_times
        
        # Get audio file path
        audio_path = join(data_dir, "ds003020/derivative/audio", f"{story}.wav")
        
        # Get embeddings for each word segment
        story_vectors = []
        for i, (word, start_time, end_time) in enumerate(zip(words, times[:-1], times[1:])):
            if word.strip() == "":
                embedding = np.zeros(768)  # wav2vec embedding size
            else:
                embedding = get_wav2vec_embeddings(audio_path, start_time, end_time)
            story_vectors.append(embedding)
        vectors[story] = np.vstack(story_vectors)
        print(f"Done {story}")

    return vectors

def downsample_wav2vec_vectors(stories, word_vectors, wordseqs):
    """Get Lanczos downsampled wav2vec vectors for specified stories.
    
    Args:
        stories: List of stories to obtain vectors for
        word_vectors: Dictionary of {story: <float32>[num_segments, vector_size]}
        wordseqs: Dictionary of DataSequence objects containing word timings
        
    Returns:
        Dictionary of {story: downsampled vectors}
    """
    downsampled_vectors = {}
    for story in stories:
        downsampled_vectors[story] = lanczosinterp2D(
            word_vectors[story], wordseqs[story].data_times, 
            wordseqs[story].tr_times, window=3)
    return downsampled_vectors

def main():
    # Define paths
    output_dir = join(DATA_DIR, "features", "wav2vec")
    os.makedirs(output_dir, exist_ok=True)

    # Get stories
    textgrid_dir = join(DATA_DIR, "ds003020/derivative/TextGrids")
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]

    # Get word sequences and wav2vec embeddings
    wordseqs = get_story_wordseqs(stories, DATA_DIR)
    vectors = get_wav2vec_vectors(stories, DATA_DIR)
    
    # Downsample vectors to TR times
    downsampled_vectors = downsample_wav2vec_vectors(stories, vectors, wordseqs)

    # Save results
    for story in stories:
        output_file = join(output_dir, f"{story}_wav2vec_embeddings.hf5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('data', data=downsampled_vectors[story])
        logging.info(f"Saved embeddings for {story}")

if __name__ == "__main__":
    main() 