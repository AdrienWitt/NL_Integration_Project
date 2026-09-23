"""
Extract TR-aligned prediction signals from a causal LM: surprisal, entropy,
and the prediction-error vector.

Three quantities, one forward pass, on exactly the grid `extract.context_lm`
already produces -- so these drop into the encoding pipeline as another
`--band NAME=STORE` beside the semantic and prosodic bands.

    surprisal(w)  = -log2 p(w | w_<t)          how violated the expectation was
    entropy(t)    = H[p(. | w_<t)]             how uncertain it was beforehand
    pe_vector(t)  = E[emb | w_<t] - emb(w)     *what* was wrong, not just how much

Why the vector and not only the scalar
--------------------------------------
The scalar is the standard surprisal regressor. The vector is the naturalistic
port of the context--statement incongruence of `../Clean_Irony`, where the
semantic band was `[H_B_hat | Delta]` with `Delta = H_B_hat - H_A` -- an
element-wise deviation of what was said from what the context set up, and the
band that produced that project's left-IFG integration result. Here the
expectation is the LM's own, so

    E[emb | w_<t] = softmax(logits_t) @ W_emb

is the probability-weighted average of every word the model expected, and the
difference against the word that actually arrived is the same object computed
from a real predictive distribution rather than from a single context vector.
One matmul on top of the logits we already need for the scalar.

Surprisal is layer-free
-----------------------
It comes off the LM head, not a hidden state, so none of the k x layer grid
that governs the semantic band applies: a k is an extraction, full stop. The
pe_vector does have a space (the input embedding matrix) but not a *choice* of
layer.

Context length is the experiment, not a nuisance
------------------------------------------------
Human expectations are built from a lossy memory of context, so the k that
best predicts a region estimates that region's effective context depth. Run
the grid and report the argmax; k=0 doubles as the lexical-frequency baseline,
since with no context the model's surprisal is close to unigram log-frequency,
which is the nuisance regressor this analysis would otherwise need from an
external corpus.

    for k in 0 4 16 64 256; do
        python -m extract.surprisal --context-words $k --pe-vector \
            --out-name surprisal_gpt2_k$k
    done
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch

from config import FEATURES_DIR, TEXTGRID_DIR, ensure_dirs
from common.ridge_utils.interpdata import lanczosinterp2D
from common.ridge_utils.story_utils import get_story_wordseqs
from extract.context_lm import BAD_WORDS, tokenize_words

log = logging.getLogger("extract.surprisal")

#: Transcription markers the TextGrids spell with braces -- `{cg}` (cough),
#: `{ig}`, `{ls}` -- which `BAD_WORDS` does not list. There are 35 of them in
#: 158,596 word slots, so for a 768-d embedding band they are noise and the
#: existing stores are unaffected. For surprisal they are the single most
#: surprising "words" in the corpus (40.7 bits against a mean of 5.6), which
#: in a one-column regressor is 6x leverage on a token that is not a word.
#: Dropped here rather than in `BAD_WORDS`, so the semantic stores keep
#: reproducing byte for byte.
NONLEXICAL = re.compile(r"^[\{\[<].*[\}\]>)]$")

#: Columns of the scalar store, in order. Written to the `columns` attribute so
#: a reader never has to count.
SCALAR_COLUMNS = ["surprisal", "entropy", "word_rate"]


@torch.no_grad()
def score_story(words: List[str], tokenizer, model, device,
                context_words: int, budget: int, stride_frac: float = 0.1,
                pe_vector: bool = False):
    """Per-word surprisal, entropy and prediction error.

    Returns `(real_idx, scalars, pe)` where `real_idx` indexes the words that
    carry tokens, `scalars` is `(n_real, 2)` of [surprisal, entropy] in bits,
    and `pe` is `(n_real, dim)` or None.

    The blocking is `embed_story`'s, for the reason given there: a causal LM
    lets word `s` into a block attend to the block's own earlier words, so a
    block filled to the token budget silently gives every word hundreds of
    words of context regardless of `k`. Capping the block at
    `stride_frac * k` bounds the real context to `[k, k(1+stride_frac)]`, and
    at k=0 the block is one word, which is the no-context baseline exactly.
    """
    word_tokens = tokenize_words(words, tokenizer)
    for i, w in enumerate(words):
        if NONLEXICAL.match(w.strip()):
            word_tokens[i] = None
    real = [i for i, t in enumerate(word_tokens) if t is not None]
    if not real:
        return [], np.zeros((0, 2), np.float32), None

    lengths = [len(word_tokens[i]) for i in real]
    bos = tokenizer.bos_token_id
    if bos is None:
        raise SystemExit(
            f"{model.config._name_or_path} has no bos_token_id. Surprisal of "
            "the first word of a block is undefined without one -- there is "
            "no position whose logits predict it.")

    emb = model.get_input_embeddings().weight if pe_vector else None
    dim = emb.shape[1] if pe_vector else 0
    scalars = np.zeros((len(real), 2), dtype=np.float32)
    pe = np.zeros((len(real), dim), dtype=np.float32) if pe_vector else None

    i, n_pass = 0, 0
    while i < len(real):
        ctx_start = max(0, i - context_words)
        used = sum(lengths[ctx_start:i]) + 1          # +1 for bos
        if used >= budget:
            while ctx_start < i and used + lengths[i] > budget:
                used -= lengths[ctx_start]
                ctx_start += 1

        stride = max(1, int(context_words * stride_frac))
        j, total = i, used
        while (j < len(real) and j - i < stride
               and total + lengths[j] <= budget):
            total += lengths[j]
            j += 1
        j = max(j, i + 1)

        ids, spans = [bos], []
        for pos in range(ctx_start, j):
            toks = word_tokens[real[pos]]
            spans.append((len(ids), len(ids) + len(toks)))
            ids.extend(toks)
        ids = ids[:budget]

        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0]

        # Only the readout words need a distribution. Slicing before the
        # softmax keeps a (seq, vocab) tensor from being materialised in
        # float32 for the whole window -- at GPT-2's 50257-word vocabulary a
        # full 1024-token block would be 206 MB per story block, for rows that
        # are context and never read.
        lo = spans[i - ctx_start][0] - 1
        hi = min(spans[j - 1 - ctx_start][1] - 1, logits.shape[0])
        if hi <= lo:
            i = j
            continue
        logp = torch.log_softmax(logits[lo:hi].float(), dim=-1)

        for pos in range(i, j):
            start, stop = spans[pos - ctx_start]
            stop = min(stop, logits.shape[0])
            if stop <= start or start - 1 < lo or stop - 1 > hi:
                continue
            rows = torch.arange(start - 1 - lo, stop - 1 - lo, device=device)
            tok = torch.tensor(ids[start:stop], device=device)
            # Sum over the word's BPE tokens: p(word) is the product of its
            # token probabilities, so surprisal is their sum. A word split
            # into more pieces is not thereby more surprising.
            nats = -logp[rows, tok].sum()
            first = logp[start - 1 - lo]
            scalars[pos] = (float(nats) / np.log(2.0),
                            float(-(first.exp() * first).sum()) / np.log(2.0))
            if pe_vector:
                expected = first.exp() @ emb.float()
                pe[pos] = (expected - emb[tok[0]].float()).cpu().numpy()

        n_pass += 1
        i = j

    log.debug(f"    {len(real)} words in {n_pass} forward passes")
    return real, scalars, pe


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--context-words", type=int, default=16,
                   help="preceding words conditioning the prediction; 0 is "
                        "the no-context (lexical frequency) baseline")
    p.add_argument("--pe-vector", action="store_true",
                   help="also write the prediction-error vector store, "
                        "E[emb|context] - emb(actual), the naturalistic port "
                        "of the irony project's incongruence vector")
    p.add_argument("--out-name", default=None)
    p.add_argument("--stories", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--stride-frac", type=float, default=0.1,
                   help="must match what extract.context_lm used for this k, "
                        "or 'surprisal at k' and 'representation at k' are "
                        "not the same k (default 0.1)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if (args.fp16 and device.type == "cuda")
        else torch.float32,
    ).to(device).eval()

    cfg = model.config
    limit = getattr(cfg, "max_position_embeddings", None) or 1024
    budget = min(args.max_tokens or limit, limit, 4096)
    if args.context_words * 1.4 > budget:
        log.warning(
            f"--context-words {args.context_words} needs roughly "
            f"{int(args.context_words * 1.4)} tokens but the window is "
            f"{budget}; context will be truncated to about "
            f"{int(budget / 1.4)} words.")

    name = args.out_name or f"surprisal_{Path(args.model).name}_k{args.context_words}"
    out_dir = Path(FEATURES_DIR) / name
    ensure_dirs(out_dir)
    pe_dir = None
    if args.pe_vector:
        pe_dir = Path(FEATURES_DIR) / name.replace("surprisal_", "pe_", 1)
        ensure_dirs(pe_dir)

    stories = ([s.strip() for s in args.stories.split(",") if s.strip()]
               if args.stories
               else sorted(q.stem for q in Path(TEXTGRID_DIR).glob("*.TextGrid")))

    log.info(f"Model  : {args.model} on {device}")
    log.info(f"Context: {args.context_words} words, {budget}-token window")
    log.info(f"Output : {out_dir}" + (f" (+ {pe_dir})" if pe_dir else ""))

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
        real, scalars, pe = score_story(
            words, tokenizer, model, device, args.context_words, budget,
            stride_frac=args.stride_frac, pe_vector=args.pe_vector)

        # Non-speech markers (breaths, pauses) carry no tokens, so their
        # surprisal is undefined -- not zero, which would read as "perfectly
        # predictable". They are dropped from the interpolation rather than
        # filled, so they contribute no time point at all.
        times = np.asarray(seq.data_times, dtype=np.float64)[real]
        tr_times = np.asarray(seq.tr_times, dtype=np.float64)

        # word_rate is an impulse per word through the same filter: the
        # nuisance every surprisal analysis needs, and free here. Without it a
        # "surprisal" effect can be a "more words in this TR" effect.
        block = np.hstack([scalars, np.ones((len(real), 1), np.float32)])
        down = lanczosinterp2D(block, times, tr_times, window=3).astype(np.float32)

        with h5py.File(out_dir / f"{story}.hf5", "w") as f:
            d = f.create_dataset("data", data=down)
            d.attrs["columns"] = np.asarray(SCALAR_COLUMNS, dtype=object)
            d.attrs["model"] = str(args.model)
            d.attrs["context_words"] = int(args.context_words)
        if pe is not None:
            pe_down = lanczosinterp2D(pe, times, tr_times,
                                      window=3).astype(np.float32)
            with h5py.File(pe_dir / f"{story}.hf5", "w") as f:
                d = f.create_dataset("data", data=pe_down)
                d.attrs["model"] = str(args.model)
                d.attrs["context_words"] = int(args.context_words)
        log.info(f"  {story}: {len(real)}/{len(words)} scored -> {down.shape}"
                 + (f" + pe {pe_down.shape}" if pe is not None else ""))

    log.info("Done.")


if __name__ == "__main__":
    main()
