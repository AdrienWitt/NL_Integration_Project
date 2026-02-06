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
from transformers import GPT2Tokenizer, GPT2Model

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

# Initialize tokenizer and model globally
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(device)  # Move model to GPU if available

def get_gpt2_embeddings(text):
    """Get GPT-2 embeddings for a given text."""
    # Tokenize and get model outputs
    inputs = tokenizer(text, return_tensors="pt")
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last hidden state embeddings and move back to CPU for numpy conversion
    embeddings = outputs.last_hidden_state.cpu().numpy()
    
    # Average across tokens to get a single embedding per text
    embeddings = np.mean(embeddings, axis=1)
    
    return embeddings

def get_gpt2_vectors(stories, data_dir):
    """Get GPT-2 embeddings for specified stories.
    
    Args:
        stories: List of stories to obtain vectors for
        data_dir: Directory containing the data
        
    Returns:
        Dictionary of {story: <float32>[num_story_words, vector_size]}
    """
    bad_words = ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", "sl"]
    wordseqs = get_story_wordseqs(stories, data_dir)
    vectors = {}
    
    for story in stories:
        # Get words and their timings
        words = wordseqs[story].data
        # Get embeddings for each word
        story_vectors = []
        for word in words:
            if word.strip() == "" or word.lower() in bad_words:
                embedding = np.zeros(768)  # GPT-2 embedding size
                print(f"Replaced '{word}' with empty embedding")
            else:
                embedding = get_gpt2_embeddings(word)
            story_vectors.append(embedding)
        vectors[story] = np.vstack(story_vectors)
        print(f"Done processing embeddings for story: {story}")

    return vectors

def downsample_gpt2_vectors(stories, word_vectors, wordseqs):
    """Get Lanczos downsampled GPT-2 vectors for specified stories.
    
    Args:
        stories: List of stories to obtain vectors for
        word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}
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
    output_dir = join(DATA_DIR, "features", "gpt2")
    os.makedirs(output_dir, exist_ok=True)

    # Get stories
    textgrid_dir = join(DATA_DIR, "ds003020/derivative/TextGrids")
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]
    stories = stories[:3]
    
    # Get word sequences and GPT-2 embeddings
    wordseqs = get_story_wordseqs(stories, DATA_DIR)
    vectors = get_gpt2_vectors(stories, DATA_DIR)
    
    # Downsample vectors to TR times
    downsampled_vectors = downsample_gpt2_vectors(stories, vectors, wordseqs)

    # Save results
    for story in stories:
        output_file = join(output_dir, f"{story}.hf5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('data', data=downsampled_vectors[story])
        logging.info(f"Saved embeddings for {story}")

if __name__ == "__main__":
    main()

