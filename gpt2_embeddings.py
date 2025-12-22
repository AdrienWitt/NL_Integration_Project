import os
import sys
import numpy as np
import h5py
import logging
import torch
from os.path import join

from transformers import GPT2Tokenizer, GPT2Model

from encoding.ridge_utils.DataSequence import DataSequence
from encoding.ridge_utils.textgrid import TextGrid
from encoding.ridge_utils.interpdata import lanczosinterp2D
from encoding.ridge_utils.story_utils import get_story_wordseqs


# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(device)
model.eval()

BAD_WORDS = [
    "sentence_start", "sentence_end", "br", "lg",
    "ls", "ns", "sp", "sl"
]


def get_embedding_dim(embedding_type):
    if embedding_type == "mean":
        return 768
    elif embedding_type in ["attention", "multi_attention"]:
        return 1536
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


# -----------------------------------------------------------------------------
# ORIGINAL GPT-2 MEAN EMBEDDINGS (UNCHANGED)
# -----------------------------------------------------------------------------

def get_gpt2_embeddings(text):
    """Mean-pooled GPT-2 embedding for a single word."""
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.cpu().numpy()
    embeddings = np.mean(embeddings, axis=1)  # [1, 768]

    return embeddings.squeeze(0).astype(np.float32)


# -----------------------------------------------------------------------------
# ATTENTION-BASED EMBEDDINGS
# -----------------------------------------------------------------------------

def get_gpt2_attention_embedding(context_text):
    """Single-head attention-weighted GPT-2 embedding."""
    inputs = tokenizer(context_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state  # [1, seq, 768]

    query = hidden[:, -1:, :]
    keys = hidden
    values = hidden

    scores = torch.matmul(query, keys.transpose(-2, -1)) / np.sqrt(768)
    weights = torch.softmax(scores, dim=-1)
    weighted = torch.matmul(weights, values).squeeze(1)
    final_vec = torch.cat([query.squeeze(1), weighted], dim=-1)

    return final_vec.cpu().numpy().astype(np.float32)


def get_gpt2_multi_attention_embedding(context_text):
    """Multi-head attention-weighted GPT-2 embedding."""
    inputs = tokenizer(context_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True
        )

    last_hidden = outputs.hidden_states[-1].squeeze(0)  # [seq, 768]
    attn = outputs.attentions[-1].squeeze(0)            # [heads, seq, seq]

    attn_to_last = attn[:, -1, :]                        # [heads, seq]
    head_contexts = torch.matmul(attn_to_last, last_hidden)# [heads, 768]
    
    current_word = last_hidden[-1]          # [768]
    context = head_contexts.mean(dim=0)     # [768]

    final_embedding = torch.cat(
    [current_word, context],
    dim=0)  

    return final_embedding.cpu().numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# GENERIC WORD-LEVEL VECTOR EXTRACTION
# -----------------------------------------------------------------------------

def get_gpt2_vectors(
    stories,
    data_dir,
    embedding_type="mean",
    context_window=10
):
    """
    embedding_type:
        - "mean"
        - "attention"
        - "multi_attention"
    """
    wordseqs = get_story_wordseqs(stories, data_dir)
    vectors = {}

    for story in stories:
        words = wordseqs[story].data
        story_vectors = []
        
        dim = get_embedding_dim(embedding_type)

        
        for i, word in enumerate(words):
            if not word.strip() or word.lower() in BAD_WORDS:
                embedding = np.zeros(dim, dtype=np.float32)

            else:
                if embedding_type == "mean":
                    embedding = get_gpt2_embeddings(word)

                else:
                    start = max(0, i - context_window)
                    context = " ".join(words[start:i + 1])

                    if embedding_type == "attention":
                        embedding = get_gpt2_attention_embedding(context)

                    elif embedding_type == "multi_attention":
                        embedding = get_gpt2_multi_attention_embedding(context)

                    else:
                        raise ValueError(f"Unknown embedding type: {embedding_type}")

            story_vectors.append(embedding)

        vectors[story] = np.vstack(story_vectors)
        logging.info(f"Finished {embedding_type} embeddings for {story}")

    return vectors


# -----------------------------------------------------------------------------
# DOWNSAMPLING
# -----------------------------------------------------------------------------

def downsample_gpt2_vectors(stories, word_vectors, wordseqs):
    """Lanczos downsampling to TR times."""
    downsampled = {}
    for story in stories:
        downsampled[story] = lanczosinterp2D(
            word_vectors[story],
            wordseqs[story].data_times,
            wordseqs[story].tr_times,
            window=3
        )
    return downsampled


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    main_dir = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project"
    textgrid_dir = "ds003020/derivative/TextGrids"
    stories = [f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]
    #stories = stories[:3]

    logging.info(f"Processing stories: {stories}")

    wordseqs = get_story_wordseqs(stories, main_dir)

    embedding_configs = {
        "gpt2_mean": dict(embedding_type="mean"),
        "gpt2_attention": dict(embedding_type="attention", context_window=10),
        "gpt2_multi_attention": dict(embedding_type="multi_attention", context_window=10),
    }

    for name, cfg in embedding_configs.items():
        logging.info(f"Starting embedding type: {name}")
        out_dir = join(main_dir, "features_new", name)
        os.makedirs(out_dir, exist_ok=True)

        vectors = get_gpt2_vectors(stories, main_dir, **cfg)
        downsampled = downsample_gpt2_vectors(stories, vectors, wordseqs)

        for story in stories:
            out_file = join(out_dir, f"{story}.hf5")
            with h5py.File(out_file, "w") as f:
                f.create_dataset("data", data=downsampled[story])

            logging.info(f"Saved {name} embeddings for {story}")


if __name__ == "__main__":
    main()
