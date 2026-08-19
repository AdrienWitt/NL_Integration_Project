# Prosody & Semantics in the LeBel dataset

Voxelwise encoding of **prosody** and **semantics** in naturalistic speech
(LeBel et al. 2023, `ds003020`), in two stages:

1. **Fine-tune** a self-supervised speech model (wav2vec2 / HuBERT / WavLM) to
   predict the 88 eGeMAPSv02 prosody functionals of each TR. Audio in, acoustics
   out — no brain data enters this stage (see "No brain data in fine-tuning").
2. **Encode**: fit ridge models from a semantic band, a prosodic band, and
   both together, then ask which voxels each modality explains and where
   combining them helps.

Extracted from `NL_Project`, restructured around one path config, one design
matrix, and one set of CV folds shared by every model.

---

## The three contrasts

All three models are fit from the **same** design matrix on the **same** folds,
which is what makes these comparisons about the features rather than about
fold assignment or preprocessing:

| Statistic | Definition | Reads as |
|---|---|---|
| `r_text`, `r_audio` | per-band prediction accuracy | semantic / prosodic tuning |
| `delta` | `r_joint − max(r_text, r_audio)` | **integration** |
| `preference` | `r_text − r_audio` | which modality dominates |
| `split_frac` | band share of the joint prediction | normalised variance contribution |

### Why banded ridge matters for `delta`

`delta` asks whether a joint model beats the better single modality. That is
only meaningful if the joint model is not handicapped relative to its
competitors.

With **one shared alpha** over a concatenated `[GPT-2 768–1536d | eGeMAPS 88d]`
design, that single alpha has to compromise between two bands with very
different dimensionality and effective SNR — while `r_text` and `r_audio` each
get their *own* optimal alpha. The joint model then loses for a purely
methodological reason: `delta` is biased downward and can go negative even in
voxels where both modalities genuinely contribute.

**Banded ridge gives every band its own regularisation**, so the joint model
properly *nests* the unimodal ones — it can shrink a band toward zero and
recover the single-modality fit. `delta` then reflects complementary
information rather than a regularisation artifact. So banded ridge is not an
alternative to the delta analysis; it is what makes the delta analysis fair.

Two consequences when reading results:

* Under banded ridge `delta ≥ 0` almost by construction, up to CV noise. The
  question is never "is it positive" but "is it above the null" — hence
  `stats/run_permutation.py`.
* The unimodal models are fit here as **single-band** banded ridge, with the
  same solver, alpha grid and folds. Anything else reintroduces the asymmetry.

The split scores are a *second* readout, not a substitute: they say how the
joint model divides its work, which is a different question from whether
joining helped. The two can disagree, and both are worth reporting.

`--backend both` additionally runs the old single-alpha solver. It
*understates* `delta` by construction, so treat it as a conservative lower
bound: a voxel significant under both is solver-independent.

---

## Audio bands, and the two fine-tuning starting points

The audio band is a swappable choice, and comparing bands is a large part of
the point. All of them are TR-aligned to the same grid and land in the same
`.hf5` format, so any one can be dropped into `--audio-features`.

| Band | Dim | From |
|---|---|---|
| `opensmile` | 88 | eGeMAPSv02 functionals — interpretable baseline |
| `emotion_avd` | 3 | audEERING arousal/dominance/valence, no fine-tuning |
| `emotion_hidden` | 1024 | the same model's pooled encoder states |
| `ft_robust` | 1024 | wav2vec2-large-robust, fine-tuned on our stories |
| `ft_emotion` | 1024 | emotion model, fine-tuned on our stories |
| `ft_robust12` | 1024 | depth-matched control, fine-tuned |

## Choosing the fine-tuning arms

`--model` takes a registry key or any Hugging Face id / fine-tuned directory;
`python -m finetune.run_finetune --list-models` prints what each key is.

### Depth is not the thing to optimise

More layers is not automatically better here.

**Capacity is not the binding constraint.** Fine-tuning runs on roughly 25k
two-second windows against a ~300M parameter encoder. Data limits the result,
not depth; extra capacity mostly buys extra overfitting risk.

**Which layer you read matters more than how many exist.** Prosodic and
paralinguistic information in wav2vec2 peaks mid-network. And the obvious
24-layer choice has a specific problem: `facebook/wav2vec2-large-960h` is
`Wav2Vec2ForCTC` with `vocab_size=32` — ASR fine-tuned, its final layers
explicitly optimised to collapse everything except which of 32 characters was
spoken. That is precisely the "how it was said" signal this project is chasing,
thrown away. Its 24 layers are real but the top third works against you.

`facebook/wav2vec2-large-robust` is `Wav2Vec2ForPreTraining` — the same 24
layers, self-supervised only, no ASR specialisation. It is also the exact base
audEERING fine-tuned from, so it is both the better generic arm and the natural
control.

### Recommended arms

| Arm | Command | Role |
|---|---|---|
| A | `--model wav2vec2-robust` | best generic 24-layer arm |
| B | `--model emotion` | the model of interest (12 layers) |
| C | `--model wav2vec2-robust --truncate-layers 12` | depth-matched control |

Run **A and B** to answer "which features predict the brain better". Add **C**
only when you want to attribute a difference to emotion pretraining rather than
to depth: A vs B confounds emotion training with depth, whereas C vs B differs
in emotion training alone.

### Which layers to retrain

Two facts set the answer, and they pull in the same direction.

**The target is low-level.** The 88 eGeMAPS functionals are F0, loudness,
jitter, shimmer, spectral slopes — acoustic descriptors whose information
already sits in the *early* layers. You are not teaching the network something
it cannot represent; you are nudging the layers you will read from.

**The dataset is small.** ~25k two-second windows against a ~300M parameter
encoder. Unfreeze too much and the network takes the shortcut: it reshapes its
whole representation to emit those 88 numbers and discards the richer structure
that made the pretrained features worth using. The result fits eGeMAPS
beautifully and predicts the brain *worse than the frozen baseline*.

So: **retrain the top half, and read from what you retrained.**

| Model | Layers | `--freeze-layers auto` | Trains | Extract after FT |
|---|---|---|---|---|
| `wav2vec2-robust` | 24 | 12 (bottom half) | 12–23 | `18-23` |
| `emotion` | 12 | 6 (bottom half) | 6–11 | `6-11` |

Always frozen regardless: the CNN feature extractor and feature projection.
Those encode low-level acoustics the pretraining already fixed, and adapting
them on this much data destabilises training.

### Sweep it — and keep an honest control

Layer choice is empirical. The sweep worth running, cheapest first:

```bash
# 0. Frozen baseline — no fine-tuning at all. THE control.
python -m extract.wav2vec --model wav2vec2-robust --layers auto --out-name base_robust

# 1. Conservative: adapt the top quarter
python -m finetune.run_finetune --model wav2vec2-robust --freeze-layers 18

# 2. Default: adapt the top half
python -m finetune.run_finetune --model wav2vec2-robust

# 3. Gentle everywhere, instead of a hard boundary
python -m finetune.run_finetune --model wav2vec2-robust \
    --freeze-layers none --llrd 0.9
```

Then compare encoding r across the four. **If arm 0 wins, fine-tuning is
hurting you** — a real possible outcome given the data size and how low-level
the target is, and worth knowing before it goes in a paper. Reporting it costs
one extraction run and no training at all.

`--llrd DECAY` is the softer alternative to a hard freeze: every layer trains,
but layer *i* gets `base_lr * decay^(n_layers-1-i)`, so with `0.9` over 24
layers the bottom moves ~10x slower than the top. It usually behaves better
than a hard boundary because there is no discontinuity in plasticity. Freezing
still wins where both apply — a frozen parameter has no gradient regardless of
its learning rate.

### Layer choice, before and after fine-tuning

The registry's `default_layers` are for the **base** models and point
mid-network. Once a model has been fine-tuned here on prosody targets, its
upper layers have been re-tuned toward prosody and the upper range becomes
right again — so extract from `18-23` on a fine-tuned 24-layer checkpoint even
though the base default is `12-17`. `--freeze-layers auto` and `--layers auto`
resolve the base defaults per model (6 of 24, 3 of 12) so depth-dependent
settings need not be remembered.

Layer choice is ultimately empirical and cheap to sweep — extract two or three
ranges under different `--out-name`s and let the encoding r decide.

The 3-d `emotion_avd` band is worth running precisely because it is tiny: if 3
affective dimensions predict a voxel about as well as 1024 learned features do,
that voxel is tracking affective prosody rather than fine acoustic detail.
Banded ridge handles the dimensionality gap properly, so the comparison is fair.

## No brain data in fine-tuning

Fine-tuning maps audio to acoustics and nothing else. An earlier design added a
second head predicting brain PCA components, on the theory that it would pull
the encoder toward the acoustic structure cortex responds to. It was removed on
2026-08-19 because it is circular: the PCA was fit on vertices **selected by
encoding r**, from the same subjects whose voxels the encoding models then
predict, so encoding scores would be inflated exactly where the effect gets
reported. The code is kept in `trash/brain_pca_multitask/`.

A non-circular version exists — fine-tune on one set of subjects, encode on a
disjoint set — but it adds a confound to the prosody-vs-semantics contrast,
which is the actual question.

## Only train stories are ever fine-tuned on

`prep/make_story_splits.py` removes the repeated story (`wheretheressmoke`)
from the pool entirely, and `finetune/run_finetune.py` **aborts** if it turns up
in either the train or the val list: the encoder would otherwise be optimised on
the audio of the story every encoding number is reported on.

The fine-tuning validation stories are a different matter — they are ordinary
training stories for the encoding stage, and using them for early stopping does
not touch the encoding test story.

## Layout

```
config.py            every path, overridable by environment variable
data/                ds003020, stimuli, features, story splits
common/              shared loaders, TR alignment, Huth ridge utilities
prep/                story splits and fine-tuning targets
finetune/            stage 1 — fine-tune the speech encoder
extract/             stimuli -> TR-aligned feature bands (.hf5)
encoding/            stage 2 — banded + single-alpha ridge
stats/               permutation testing and contrast maps
results/             outputs (git-ignored)
scripts/             end-to-end driver
```

## Install

```bash
pip install -r requirements.txt
python -m config          # prints every resolved path and whether it exists
```

`config.py` is the single source of truth. Any path can be overridden without
editing it, which is how the same code runs on a laptop and on a cluster:

```bash
export FMRI_DIR=/scratch/$USER/ds003020/derivative/preprocessed_data
```

## Pipeline

```bash
# 1. Story splits
python -m prep.make_story_splits          # train / val / held-out for fine-tuning
python -m prep.make_encoding_splits       # subject -> stories, for encoding

# 2. Feature bands
python -m extract.opensmile --out-name opensmile      # 88 eGeMAPS, prosody baseline
python -m extract.gpt2 --type mean                    # semantic band
python -m extract.emotion_avd --output avd            # 3-d arousal/dominance/valence

# 3. Fine-tuning targets (88 eGeMAPS per TR; audio only, no fMRI needed)
python -m prep.make_finetune_targets

# 4. Fine-tune — same train stories, three arms (see "Choosing the arms")
python -m finetune.run_finetune --list-models
python -m finetune.run_finetune --model wav2vec2-robust
python -m finetune.run_finetune --model emotion
python -m finetune.run_finetune --model wav2vec2-robust --truncate-layers 12
                               # optional control; auto freezes 6 of the 12 kept

# 5. Features from each fine-tuned encoder (--layers explicit for local dirs)
python -m extract.wav2vec \
    --model results/finetune/wav2vec2_robust_frozen_12_lr3e-05_seed42/final_model \
    --layers 18-23 --out-name ft_robust
python -m extract.wav2vec \
    --model results/finetune/emotion_frozen_6_lr3e-05_seed42/final_model \
    --layers 6-11 --out-name ft_emotion

# 6. Encoding
python -m encoding.run_encoding --subjects all \
    --text-features gpt2_mean --audio-features wav2vec_ft_18to23 \
    --backend both --eval holdout --min-ev 0.1

# 7. Statistics
python -m stats.run_permutation --subjects all \
    --text-features gpt2_mean --audio-features wav2vec_ft_18to23 --n-perms 1000
python -m stats.analysis --features gpt2_mean__wav2vec_ft_18to23
```

`scripts/run_pipeline.sh` runs all of it.

### Evaluation modes

* `--eval holdout` (default) — fit on every training story, score on the
  repeated story `wheretheressmoke`. Its repeats give an explainable-variance
  ceiling, so scores can be expressed as a fraction of what is achievable
  (`stats.analysis --normalize`).
* `--eval cv` — nested CV over the training stories, for subjects without the
  repeated story. Hyperparameters are re-selected inside each outer fold, so
  no held-out story influences the alphas used to predict it.

---

## Things that will bite you

**Fine-tuning targets are the 88 eGeMAPSv02 functionals.** The dataset refuses
to build if the targets have a different width (`--expect-n-features 0`
disables the check). A silent mismatch here trains the model against the wrong
target set and looks like nothing worse than mediocre metrics.

**Feature/response TR alignment is checked, not assumed.** Features live on a
grid of `respdict[story] − TR_PAD` onsets; stored responses may be on that same
grid or on the raw acquisition grid. `preprocess.trim_response` anchors the two
at their common end, applies the identical trim, logs which offset it found,
and raises on anything else. A wrong offset shifts every label by a fixed
number of TRs and degrades results without ever looking broken.

**Fine-tuned checkpoints must be unwrapped.** `train_model` saves an
`AudioEncoderForProsody`, whose weights are nested
under an `encoder.` prefix. Loading such a directory straight into
`Wav2Vec2Model` does not line those keys up — the fine-tuned weights are
dropped and randomly initialised ones silently replace them, so the
"fine-tuned" features are nothing of the sort. `extract/wav2vec.py` detects our
checkpoints from their config and unwraps `.encoder` explicitly.

**Validation scalers come from training.** `ProsodyBrainDataset.get_fitted_scalers()`
is passed to the validation set; refitting there leaks the validation
distribution into the targets and inflates the metrics.

**OneDrive on-demand files.** Much of `data/` is cloud-backed. Files that have
not been hydrated raise `OSError: [Errno 5]` on open. Mark the directories
"Always keep on this device", or point `FMRI_DIR` at local storage, before a
long run.

**`--min-ev 0.1` is worth using.** It restricts fitting to voxels with real
stimulus-locked signal, cuts runtime substantially, and results are scattered
back into the full voxel space so every saved map stays comparable.

---

## Data

Data was **moved** here out of `NL_Project` — `data/ds003020`, `data/stimuli_16k`,
`data/features`, `data/derivative`, plus the previous results and fine-tuning
checkpoints under `results/`. `NL_Project` keeps its code and git history but no
longer holds the data.

## Attribution

`common/ridge_utils/` and `encoding/huth_ridge.py` adapt the Huth lab's
[deep-fMRI-dataset](https://github.com/HuthLab/deep-fMRI-dataset). The banded
ridge pipeline follows the
[voxelwise_tutorials](https://gallantlab.org/voxelwise_tutorials/) approach
using [himalaya](https://github.com/gallantlab/himalaya). Dataset: LeBel et al.
(2023), *A natural language fMRI dataset for voxelwise encoding models*,
`ds003020`.
