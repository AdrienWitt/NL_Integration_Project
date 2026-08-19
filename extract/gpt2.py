"""
Extract the TR-aligned GPT-2 semantic band from the story transcripts.

Words come from the ds003020 TextGrids with their onset times, are embedded
one at a time, then Lanczos-interpolated onto the TR grid — words do not land
on TR boundaries, and interpolating is what puts a word-rate signal onto the
2 s sampling grid without aliasing.

Embedding types
---------------
``mean``
    Mean-pooled GPT-2 states of the word alone. No context, so it is close to
    a static word vector: a useful lower baseline.
``attention``
    The word's final state concatenated with a single-head attention-weighted
    summary of the preceding `--context-window` words. 1536-d.
``multi_attention``
    Same idea, but the context summary averages the last layer's attention
    heads. 1536-d.

Context matters here: a purely lexical band would understate how much of the
"semantic" signal is really contextual integration, which is exactly the
comparison the prosody/semantics contrast rests on.

Usage
-----
    python -m extract.gpt2 --type mean
    python -m extract.gpt2 --type attention --context-window 10
"""

import argparse
import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch

from config import FEATURES_DIR, TEXTGRID_DIR, ensure_dirs
from common.ridge_utils.interpdata import lanczosinterp2D
from common.ridge_utils.story_utils import get_story_wordseqs

log = logging.getLogger("extract.gpt2")

#: TextGrid tokens that mark non-speech events rather than words.
BAD_WORDS = {"sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", "sl"}

EMBEDDING_DIM = {"mean": 768, "attention": 1536, "multi_attention": 1536}


@torch.no_grad()
def embed_mean(text: str, tokenizer, model, device) -> np.ndarray:
    inputs = {k: v.to(device) for k, v in
              tokenizer(text, return_tensors="pt").items()}
    hidden = model(**inputs).last_hidden_state
    return hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_attention(context: str, tokenizer, model, device) -> np.ndarray:
    """Last word's state + its single-head attention summary of the context."""
    inputs = {k: v.to(device) for k, v in
              tokenizer(context, return_tensors="pt").items()}
    hidden = model(**inputs).last_hidden_state          # [1, seq, 768]

    query = hidden[:, -1:, :]
    scores = torch.matmul(query, hidden.transpose(-2, -1)) / np.sqrt(hidden.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    summary = torch.matmul(weights, hidden).squeeze(1)
    return torch.cat([query.squeeze(1), summary], dim=-1).squeeze(0) \
        .cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_multi_attention(context: str, tokenizer, model, device) -> np.ndarray:
    """Last word's state + the head-averaged attention summary of the context."""
    inputs = {k: v.to(device) for k, v in
              tokenizer(context, return_tensors="pt").items()}
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    last_hidden = outputs.hidden_states[-1].squeeze(0)   # [seq, 768]
    attn = outputs.attentions[-1].squeeze(0)             # [heads, seq, seq]

    attn_to_last = attn[:, -1, :]                        # [heads, seq]
    head_contexts = torch.matmul(attn_to_last, last_hidden)  # [heads, 768]

    return torch.cat([last_hidden[-1], head_contexts.mean(dim=0)], dim=0) \
        .cpu().numpy().astype(np.float32)


def story_word_vectors(words: List[str], kind: str, context_window: int,
                       tokenizer, model, device) -> np.ndarray:
    dim = EMBEDDING_DIM[kind]
    vectors = []
    for i, word in enumerate(words):
        if not word.strip() or word.lower() in BAD_WORDS:
            # Non-speech markers get a zero vector so the word/time alignment
            # is preserved; dropping them would shift every later onset.
            vectors.append(np.zeros(dim, dtype=np.float32))
            continue

        if kind == "mean":
            vectors.append(embed_mean(word, tokenizer, model, device))
            continue

        context = " ".join(words[max(0, i - context_window): i + 1])
        if kind == "attention":
            vectors.append(embed_attention(context, tokenizer, model, device))
        else:
            vectors.append(embed_multi_attention(context, tokenizer, model, device))

    return np.vstack(vectors)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--type", default="mean", choices=sorted(EMBEDDING_DIM),
                   dest="kind")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--context-window", type=int, default=10)
    p.add_argument("--out-name", default=None,
                   help="default: gpt2_<type>")
    p.add_argument("--stories", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    from transformers import GPT2Model, GPT2Tokenizer

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model).to(device).eval()

    out_dir = Path(FEATURES_DIR) / (args.out_name or f"gpt2_{args.kind}")
    ensure_dirs(out_dir)

    if args.stories:
        stories = [s.strip() for s in args.stories.split(",") if s.strip()]
    else:
        stories = sorted(p.stem for p in Path(TEXTGRID_DIR).glob("*.TextGrid"))

    log.info(f"Model  : {args.model} ({args.kind}, "
             f"{EMBEDDING_DIM[args.kind]}-d) on {device}")
    log.info(f"Output : {out_dir}")
    log.info(f"{len(stories)} stories")

    pending = [s for s in stories
               if args.overwrite or not (out_dir / f"{s}.hf5").exists()]
    if not pending:
        log.info("Everything already extracted; nothing to do.")
        return

    # Word sequences carry the TR grid each story is interpolated onto.
    wordseqs = get_story_wordseqs(pending)

    for story in pending:
        seq = wordseqs[story]
        vectors = story_word_vectors(list(seq.data), args.kind,
                                     args.context_window, tokenizer, model, device)
        downsampled = lanczosinterp2D(vectors, seq.data_times, seq.tr_times,
                                      window=3).astype(np.float32)

        out_path = out_dir / f"{story}.hf5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=downsampled)
        log.info(f"  {story}: {len(seq.data)} words -> {downsampled.shape}")

    log.info("Done.")


if __name__ == "__main__":
    main()
