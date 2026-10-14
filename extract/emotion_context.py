"""
Emotion band with acoustic context and a choice of temporal resolution.

The emotion band of the paper (`perlayer_base_emotion`) passes each TR's 2-s
window, alone, through the audEERING emotion model and mean-pools it. The text
band, by contrast, sees hundreds of preceding words and is resampled word by
word. This script removes both asymmetries so that they can be swept:

    --readout R   seconds read out per step. R = 2 reproduces the paper's grid
                  (one step per TR, mean over the TR's frames). R = 0.5 steps
                  every 0.5 s and the steps are resampled to the TR grid with
                  the same Lanczos filter as the text band.
    --context W   seconds of preceding audio given to the model before each
                  readout segment. The model attends over context + readout,
                  but only the readout frames are pooled -- the same rule as the
                  text band, where context words feed attention and only the
                  current words are read out. Mean-pooling the whole window
                  instead would dilute the present in the past.

W = 0, R = 2 is the paper's band; the first test of this script is that it
reproduces `perlayer_base_emotion` to float precision.

Context never reaches before the start of the sound (no silence is fed as
context); readout segments that fall outside the sound are zero-padded as in
`extract.wav2vec`, so those TRs stay on the same grid and are trimmed later.

Two stores per (R, W):
    <out>/            (n_TRs, 12, 1024) hidden states of transformer blocks
                      0..11, same layout and attrs as perlayer_base_emotion
    <out>_avd/        (n_TRs, 3) the regression head (arousal, dominance,
                      valence) applied to the pooled readout frames of the
                      last layer, as the model's own forward does over its input

    python -m extract.emotion_context --readout 0.5 --context 8 \\
        --stories-json data/derivative/common_stories_all9.json
"""

import argparse
import json
import logging
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from config import FEATURES_DIR, SAMPLING_RATE, STIMULI_16K_DIR, TR, ensure_dirs
from common.tr_alignment import load_trfiles, tr_onsets, window_bounds
from common.ridge_utils.interpdata import lanczosinterp2D
from extract.emotion_avd import EMOTION_MODEL, build_emotion_model
from extract.wav2vec import load_waveform

log = logging.getLogger("extract.emotion_context")

#: wav2vec2 feature encoder: 400-sample receptive field, 320-sample stride.
FRAME_STRIDE = 320


def readout_starts(onsets: np.ndarray, readout: float) -> np.ndarray:
    """Start times (s, sound clock) of the readout segments.

    R = TR gives exactly the TR onsets (the paper's grid). Shorter readouts
    tile the same span, from the first TR onset to the end of the last TR.
    """
    if math.isclose(readout, TR):
        return onsets.astype(float)
    n = int(round((onsets[-1] + TR - onsets[0]) / readout))
    return onsets[0] + readout * np.arange(n)


@torch.no_grad()
def embed_segment(waveform: torch.Tensor, start: float, readout: float,
                  context: float, processor, model, device):
    """Hidden states (12, 1024) and AVD (3,) for one readout segment."""
    r_samp = int(round(readout * SAMPLING_RATE))
    lo, hi, pad_l, pad_r = window_bounds(start, SAMPLING_RATE, r_samp,
                                         waveform.shape[1])
    seg = torch.nn.functional.pad(waveform[:, lo:hi], (pad_l, pad_r))
    if context > 0 and start > 0:
        c_lo = max(0, int(round((start - context) * SAMPLING_RATE)))
        seg = torch.cat([waveform[:, c_lo:lo], seg], dim=1)
    audio = seg.squeeze(0).numpy()

    inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    x = inputs.input_values.to(device)
    out = model.wav2vec2(x, output_hidden_states=True)
    n_frames = out.hidden_states[0].shape[1]
    # frames whose window starts inside the readout segment
    k0 = min(int(math.ceil((x.shape[1] - r_samp) / FRAME_STRIDE)), n_frames - 1)
    # hidden_states[0] is the CNN output; block i is at i + 1 (as in extract.wav2vec)
    hid = torch.stack([out.hidden_states[i + 1][0, k0:].mean(0) for i in range(12)])
    avd = model.classifier(out.last_hidden_state[0, k0:].mean(0, keepdim=True))[0]
    return hid.float().cpu().numpy(), avd.float().cpu().numpy()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readout", type=float, required=True, help="seconds (2 or 0.5)")
    p.add_argument("--context", type=float, required=True, help="seconds of preceding audio")
    p.add_argument("--model", default=EMOTION_MODEL)
    p.add_argument("--audio-dir", default=str(STIMULI_16K_DIR))
    p.add_argument("--stories-json", default=None,
                   help="json with a 'common_stories' list (default: every wav)")
    p.add_argument("--stories", default=None, help="comma-separated subset")
    p.add_argument("--out-name", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    r_tag = str(a.readout).replace(".", "p").rstrip("0").rstrip("p")
    out_name = a.out_name or f"perlayer_emoctx_r{r_tag}_w{a.context:g}"
    out_dir, avd_dir = Path(FEATURES_DIR) / out_name, Path(FEATURES_DIR) / f"{out_name}_avd"
    ensure_dirs(out_dir)
    ensure_dirs(avd_dir)

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from transformers import AutoFeatureExtractor
    processor = AutoFeatureExtractor.from_pretrained(a.model)
    model = build_emotion_model(a.model).to(device).eval()
    assert model.config.num_hidden_layers == 12, model.config.num_hidden_layers

    trfiles = load_trfiles()
    audio_dir = Path(a.audio_dir)
    if a.stories:
        stories = [s.strip() for s in a.stories.split(",") if s.strip()]
    elif a.stories_json:
        stories = json.load(open(a.stories_json))["common_stories"]
    else:
        stories = sorted(q.stem for q in audio_dir.glob("*.wav"))
    log.info(f"readout {a.readout} s, context {a.context} s, {len(stories)} stories "
             f"-> {out_dir.name} on {device}")

    for story in stories:
        out_path = out_dir / f"{story}.hf5"
        if out_path.exists() and not a.overwrite:
            continue
        waveform = load_waveform(audio_dir / f"{story}.wav")
        onsets = tr_onsets(story, trfiles)
        starts = readout_starts(onsets, a.readout)
        hid, avd = zip(*(embed_segment(waveform, s, a.readout, a.context,
                                       processor, model, device) for s in starts))
        hid, avd = np.stack(hid), np.stack(avd)
        if not math.isclose(a.readout, TR):
            # same filter and target grid (TR centres) as the text band
            # Huth's lanczosinterp2D sums its weights rather than averaging
            # (gain ~4 at R = 0.5 s); divide by the weight sum so the values
            # keep the model's own scale, e.g. AVD stays on the head's range.
            centres, tr_times = starts + a.readout / 2, onsets + TR / 2
            gain = lanczosinterp2D(np.ones((len(starts), 1)), centres, tr_times,
                                   window=3)
            hid = (lanczosinterp2D(hid.reshape(len(starts), -1), centres, tr_times,
                                   window=3) / gain).reshape(len(onsets), 12, -1)
            avd = lanczosinterp2D(avd, centres, tr_times, window=3) / gain
        for path, data, extra in ((out_path, hid, True), (avd_dir / f"{story}.hf5", avd, False)):
            with h5py.File(path, "w") as f:
                d = f.create_dataset("data", data=data.astype(np.float32))
                if extra:
                    d.attrs["layers"] = np.arange(12, dtype=np.int32)
                    d.attrs["base_model"] = str(a.model)
                d.attrs["readout_sec"] = float(a.readout)
                d.attrs["context_sec"] = float(a.context)
        log.info(f"  {story}: {len(starts)} steps -> {hid.shape}")
    log.info("done")


if __name__ == "__main__":
    main()
