import os
import numpy as np
import h5py
import torch
from transformers import GPT2Tokenizer, GPT2Model
from encoding.ridge_utils.DataSequence import DataSequence
from encoding.ridge_utils.textgrid import TextGrid
from encoding.ridge_utils.interpdata import lanczosinterp2D
from encoding.ridge_utils.stimulus_utils import load_textgrids
from encoding.ridge_utils.dsutils import make_word_ds
from encoding.ridge_utils.story_utils import get_story_wordseqs
import logging
from os.path import join

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Initialize tokenizer and model globally
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(device)

def get_gpt2_attention_embeddings(context_text):
    """Get attention-weighted GPT-2 embedding for a context."""
    # Tokenize - EXACTLY like working script
    inputs = tokenizer(context_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    context_embeddings = outputs.last_hidden_state  # Keep as PyTorch tensor [1, seq_len, 768]
    
    # Apply scaled dot-product attention 
    embedding_dim = 768
    query = context_embeddings[:, -1:, :]  # Last word as query [1, 1, 768]
    keys = context_embeddings                   # [1, seq_len, 768]
    values = context_embeddings                 # [1, seq_len, 768]
    
    attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / np.sqrt(embedding_dim)
    attention_weights = torch.softmax(attention_scores, dim=-1)  # [1, 1, seq_len]
    weighted_embedding = torch.matmul(attention_weights, values).squeeze(1)  # [1, 768]
    
    return weighted_embedding.cpu().numpy()

def get_gpt2_attention_vectors(stories, main_dir, context_window=10):
    """Get GPT-2 attention embeddings for specified stories."""
    bad_words = ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", "sl"]
    wordseqs = get_story_wordseqs(stories, main_dir)
    vectors = {}
    
    for story in stories:
        words = wordseqs[story].data
        story_vectors = []
        
        for i in range(len(words)):
            start_idx = max(0, i - context_window)
            context_words = words[start_idx:i + 1]
            context_text = " ".join(context_words)
            
            # FIXED: Only filter CURRENT WORD, not context!
            if not words[i].strip() or words[i].lower() in bad_words:  # ← CHANGED
                embedding = np.zeros(768)
                print(f"Replaced WORD '{words[i]}' with empty embedding")
            else:
                embedding = get_gpt2_multi_attention_embeddings(context_text)  # ALL context!
            
            story_vectors.append(embedding)
        
        vectors[story] = np.vstack(story_vectors).astype(np.float32)
        print(f"Done processing attention embeddings for story: {story}")

    return vectors


def get_gpt2_multi_attention_embeddings(context_text):
    """NEW: Multi-head attention-weighted GPT-2 embedding (current word = query)."""
    inputs = tokenizer(context_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
   
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True
        )
   
    # [1, seq_len, 768]
    last_layer_states = outputs.hidden_states[-1]  # [1, seq, 768]
    seq_len = last_layer_states.size(1)
    
    # [1, num_heads, seq, seq] → [num_heads, seq, seq]
    attn_maps = outputs.attentions[-1]      # last layer attention
    attn_maps = attn_maps.squeeze(0)        # [num_heads, seq, seq]
    
    # Extract attention from **current (last) token** to all previous tokens
    # attn_to_current: [num_heads, seq] — how much each position attends TO the last token
    attn_to_current = attn_maps[:, -1, :]   # [num_heads, seq]
    
    # Values: all token embeddings in the sequence
    values = last_layer_states.squeeze(0)   # [seq, 768]
    
    # Compute per-head context vector: sum over source positions
    # (num_heads, seq) @ (seq, 768) → (num_heads, 768)
    head_contexts = torch.matmul(attn_to_current, values)  # [num_heads, 768]
    
    # Average over heads → final embedding
    weighted_embedding = head_contexts.mean(dim=0)  # [768]
    
    return weighted_embedding.cpu().numpy().astype(np.float32)

def downsample_gpt2_vectors(stories, word_vectors, wordseqs):
    """Get Lanczos downsampled GPT-2 vectors for specified stories."""
    downsampled_vectors = {}
    for story in stories:
        downsampled_vectors[story] = lanczosinterp2D(
            word_vectors[story], wordseqs[story].data_times, 
            wordseqs[story].tr_times, window=3)
    return downsampled_vectors

def main():
    # Define paths
    main_dir = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project"
    output_dir = join(main_dir, "features", "gpt2_multi_attention")
    os.makedirs(output_dir, exist_ok=True)

    # Get all stories from TextGrids directory
    textgrid_dir = "ds003020/derivative/TextGrids"
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]
    logging.info(f"Found {len(stories)} stories: {stories}")

    # Get word sequences and GPT-2 attention embeddings
    wordseqs = get_story_wordseqs(stories, main_dir)
    vectors = get_gpt2_attention_vectors(stories, main_dir, context_window=10)
    
    # Downsample vectors to TR times
    downsampled_vectors = downsample_gpt2_vectors(stories, vectors, wordseqs)

    # Save results as {story}.hf5
    for story in stories:
        output_file = join(output_dir, f"{story}.hf5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('data', data=downsampled_vectors[story])
        logging.info(f"Saved attention embeddings for {story}")

if __name__ == "__main__":
    main()