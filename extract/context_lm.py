"""
Extract a TR-aligned semantic band with real linguistic context.

Each word is embedded *in context*: the preceding `--context-words` words are
prepended, the sentence is run through a causal LM once, and the word's own
token states are read out. Word vectors are then Lanczos-interpolated onto the
TR grid, exactly as `extract.gpt2` does — so this drops into the existing
encoding pipeline as another `--text-features <dir>`.

Why this and not `extract.gpt2 --type mean`
-------------------------------------------
`extract.gpt2 --type mean` embeds each word *alone*, which is close to a static
word vector. Everything downstream then compares a lookup table against a
1024-d contextual audio band. Here word 3 of a TR sees words 1-2, and the TR
sees everything before it, up to `--context-words` — both senses of "context"
you get for free from a causal LM, without touching the TR alignment.

This is not the same thing as `--ndelays`. The FIR delays model how the *BOLD
response* depends on the previous few TRs; this changes what the *features
themselves* represent. You want both.

Context length is a result, not just a hyperparameter (Jain & Huth 2018;
TRIBE fig. 6c reports no plateau even at 1024 words), so sweep it before
swapping in a bigger model — otherwise a gain confounds context with scale.

    # the sweep: k=0 reproduces the current gpt2_mean baseline
    for k in 0 1 4 16 64 256; do
        python -m extract.context_lm --model gpt2 --context-words $k \
            --out-name gpt2_k$k
    done

    # then, at the winning k, the model swap
    python -m extract.context_lm --model meta-llama/Llama-3.2-3B \
        --context-words 1024 --layers 0.75 --out-name llama3b_k1024_f75

Layers
------
`--layers` takes integers (`18`), a range (`18-23`, averaged), or fractions of
depth (`0.75`), which is how TRIBE specifies them and the only way to compare
a 12-layer GPT-2 with a 28-layer Llama. Prefer a single layer: on the audio
side, averaged ranges never beat the best single layer here.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

from config import FEATURES_DIR, TEXTGRID_DIR, ensure_dirs
from common.ridge_utils.interpdata import lanczosinterp2D
from common.ridge_utils.story_utils import get_story_wordseqs

log = logging.getLogger("extract.context_lm")

#: TextGrid tokens marking non-speech events rather than words. They keep
#: their slot in the time series (as a zero vector, so onsets do not shift)
#: but contribute no tokens to the language model's context.
BAD_WORDS = {"sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", "sl"}


def parse_layers(spec: str, n_layers: int) -> List[int]:
    """'last' -> [n]; '18' -> [18]; '18-23' -> [18..23]; '0.75' -> [round(.75n)].

    `n_layers` counts transformer blocks; hidden_states has n_layers + 1
    entries, index 0 being the embeddings, so a fraction of 1.0 is the last
    block and 0.0 is the embedding layer.
    """
    spec = spec.strip().lower()
    if spec in ("last", "final"):
        return [n_layers]
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "." in part:
            frac = float(part)
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"layer fraction {part} outside [0, 1]")
            out.append(int(round(frac * n_layers)))
        else:
            out.append(int(part))
    return out


def tokenize_words(words: List[str], tokenizer) -> List[Optional[List[int]]]:
    """Token ids per word, or None for non-speech markers.

    A leading space is prepended to every word: BPE tokenizers encode
    " word" and "word" as different tokens, and the space-prefixed form is
    what the model actually sees in running text.
    """
    ids = []
    for word in words:
        clean = word.strip()
        if not clean or clean.lower() in BAD_WORDS:
            ids.append(None)
            continue
        toks = tokenizer.encode(" " + clean, add_special_tokens=False)
        ids.append(toks or None)
    return ids


@torch.no_grad()
def embed_story(words: List[str], tokenizer, model, device, layers: List[int],
                context_words: int, budget: int, dim: int,
                stride_frac: float = 0.1) -> np.ndarray:
    """One contextual vector per word, in the order given.

    Words are emitted in blocks: one forward pass covers `context_words` words
    of context plus a block of new words, and only the new words are read out.

    The block size is what makes this correct rather than merely fast. In a
    causal LM every word in the block also attends to the earlier words *of
    the block*, so a word `s` positions into a block sees `k + s` words, not
    `k`. Filling the block to the token budget therefore gives every word
    hundreds of words of context no matter what `k` says -- which silently
    collapses a context-length sweep, the one measurement this script exists
    to support. Capping the block at `stride_frac * k` bounds the actual
    context to [k, k(1+stride_frac)], and at k=0 the block is one word, which
    is the isolated-word baseline exactly.
    """
    word_tokens = tokenize_words(words, tokenizer)
    real = [i for i, t in enumerate(word_tokens) if t is not None]
    vectors = np.zeros((len(words), dim), dtype=np.float32)
    if not real:
        return vectors

    lengths = [len(word_tokens[i]) for i in real]
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None

    i = 0
    n_pass = 0
    while i < len(real):
        ctx_start = max(0, i - context_words)
        used = sum(lengths[ctx_start:i]) + (1 if bos is not None else 0)
        if used >= budget:
            # Context alone fills the window; trim it from the left so at
            # least one new word fits. Only bites when context_words is large
            # relative to the model's positions (GPT-2 tops out at 1024).
            while ctx_start < i and used + lengths[i] > budget:
                used -= lengths[ctx_start]
                ctx_start += 1

        stride = max(1, int(context_words * stride_frac))
        j, total = i, used
        while (j < len(real) and j - i < stride
               and total + lengths[j] <= budget):
            total += lengths[j]
            j += 1
        j = max(j, i + 1)  # always make progress, even on a monster token

        ids, spans = ([bos] if bos is not None else []), []
        for pos in range(ctx_start, j):
            toks = word_tokens[real[pos]]
            spans.append((len(ids), len(ids) + len(toks)))
            ids.extend(toks)
        ids = ids[:budget]

        hidden = model(input_ids=torch.tensor([ids], device=device),
                       output_hidden_states=True).hidden_states
        stack = torch.stack([hidden[layer][0] for layer in layers]).mean(0)

        for pos in range(i, j):
            start, stop = spans[pos - ctx_start]
            stop = min(stop, stack.shape[0])
            if stop <= start:
                continue
            vectors[real[pos]] = stack[start:stop].mean(0).float().cpu().numpy()

        n_pass += 1
        i = j

    log.debug(f"    {len(real)} words in {n_pass} forward passes")
    return vectors


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2",
                   help="any causal LM on the hub, e.g. gpt2, gpt2-xl, "
                        "meta-llama/Llama-3.2-3B")
    p.add_argument("--context-words", type=int, default=256,
                   help="preceding words prepended to each word (0 reproduces "
                        "the isolated-word baseline)")
    p.add_argument("--layers", default="last",
                   help="'last', '18', '18-23' (averaged), or a fraction of "
                        "depth such as 0.75")
    p.add_argument("--out-name", default=None)
    p.add_argument("--stories", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--fp16", action="store_true",
                   help="half precision; worth it for models above ~1B")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap the forward-pass window (default: the model's own "
                        "limit, capped at 4096 for memory)")
    p.add_argument("--stride-frac", type=float, default=0.1,
                   help="words read out per pass, as a fraction of "
                        "--context-words. Actual context lands in "
                        "[k, k(1+frac)]; lower is more exact and slower "
                        "(default 0.1)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if (args.fp16 and device.type == "cuda")
        else torch.float32,
    ).to(device).eval()

    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    dim = getattr(cfg, "hidden_size", None) or cfg.n_embd
    layers = parse_layers(args.layers, n_layers)
    if max(layers) > n_layers or min(layers) < 0:
        raise SystemExit(f"--layers {args.layers} -> {layers}, outside "
                         f"0..{n_layers} for {args.model}")

    limit = getattr(cfg, "max_position_embeddings", None) or 1024
    budget = min(args.max_tokens or limit, limit, 4096)
    # GPT-2 stops at 1024 *tokens*, so a 1024-*word* context cannot fit: the
    # loop trims from the left and the effective context is shorter than
    # asked. Say so rather than reporting a context length that never applied.
    if args.context_words * 1.4 > budget:
        log.warning(
            f"--context-words {args.context_words} needs roughly "
            f"{int(args.context_words * 1.4)} tokens but the window is "
            f"{budget}; context will be truncated to about "
            f"{int(budget / 1.4)} words. Use a longer-context model to go "
            f"further."
        )

    out_dir = Path(FEATURES_DIR) / (
        args.out_name or f"{Path(args.model).name}_k{args.context_words}")
    ensure_dirs(out_dir)

    stories = ([s.strip() for s in args.stories.split(",") if s.strip()]
               if args.stories
               else sorted(q.stem for q in Path(TEXTGRID_DIR).glob("*.TextGrid")))

    log.info(f"Model  : {args.model} ({n_layers} layers, {dim}-d) on {device}")
    log.info(f"Layers : {args.layers} -> {layers}")
    log.info(f"Context: {args.context_words} words, {budget}-token window")
    log.info(f"Output : {out_dir}")

    pending = [s for s in stories
               if args.overwrite or not (out_dir / f"{s}.hf5").exists()]
    if not pending:
        log.info("Everything already extracted; nothing to do.")
        return
    log.info(f"{len(pending)}/{len(stories)} stories to do")

    wordseqs = get_story_wordseqs(pending)
    for story in pending:
        seq = wordseqs[story]
        words = list(seq.data)
        vectors = embed_story(words, tokenizer, model, device, layers,
                              args.context_words, budget, dim,
                              stride_frac=args.stride_frac)
        downsampled = lanczosinterp2D(vectors, seq.data_times, seq.tr_times,
                                      window=3).astype(np.float32)
        with h5py.File(out_dir / f"{story}.hf5", "w") as f:
            f.create_dataset("data", data=downsampled)
        log.info(f"  {story}: {len(words)} words -> {downsampled.shape}")

    log.info("Done.")


if __name__ == "__main__":
    main()
