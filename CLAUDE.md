# Prosody_Semantics_NL — project context

Voxelwise encoding of prosody and semantics in the LeBel `ds003020` dataset.
Created 2026-08-18 by extracting the useful parts of `../NL_Project`.

Read `README.md` first — it carries the full rationale. This file is the
short version plus the decisions and their reasons, so a fresh session does not
re-litigate settled questions.

## What the project does

1. **Fine-tune** a speech encoder (wav2vec2 / HuBERT / WavLM / an emotion model)
   to predict the 88 eGeMAPSv02 prosody functionals per TR. Audio in,
   acoustics out — no brain data enters this stage.
2. **Encode**: fit a semantic band, a prosodic band, and both jointly, then ask
   which voxels each modality explains and where combining them helps.

Headline statistics: `delta = r_joint − max(r_text, r_audio)` (integration),
`preference = r_text − r_audio`, and banded-ridge split scores.

## Settled decisions — do not re-open without reason

**Banded ridge is primary; single-alpha ridge is a cross-check.**
`delta` is only fair if the joint model is not handicapped. One shared alpha
over `[GPT-2 768–1536d | audio 88–1024d]` must compromise between bands of very
different dimensionality while the unimodal models each get their own optimal
alpha — biasing `delta` downward, sometimes negative. Per-band alphas make the
joint model properly nest the unimodal ones. `--backend both` also runs the old
solver; it understates `delta` by construction, so it is a conservative lower
bound, not a second opinion of equal weight.
Corollary: under banded ridge `delta ≥ 0` almost by construction, so the test is
always "significantly above the permutation null", never "is it positive".

**All three models share one design matrix and one set of CV folds.** That is
what makes the contrasts about features rather than fold assignment.

**No brain data in fine-tuning (decided 2026-08-19).** A multi-task head
predicting brain PCA components was removed as circular: the PCA was fit on
fsaverage vertices *selected by encoding r*, from the same subjects whose voxels
the encoding models then predict, so encoding scores would be inflated exactly
where the effect is reported. Removed code lives in `trash/brain_pca_multitask/`
(untracked). Do not reinstate without a subject-disjoint design — and note the
old objective was also mis-scaled: PC variances measured `[869, 98, 90]` against
z-scored prosody at ~1, so `--brain-weight` was meaningless.
Consequence: `prep/make_finetune_targets.py` needs no fMRI data at all, so
`FSAVERAGE_DIR` being MISSING no longer blocks fine-tuning.

**Fine-tuning never sees the encoding test story.** `wheretheressmoke` (the
repeated story, hence the only explainable-variance ceiling) is dropped from the
fine-tuning pool, and `finetune/run_finetune.py` aborts if it appears in train
or val — the encoder would otherwise be optimised on the audio of the one story
every encoding number is reported on.

**Freeze the bottom half, adapt the top half.** Derived from depth in
`finetune/registry.py` (12 of 24; 6 of 12), CNN front end always frozen.
The target (eGeMAPS) is low-level and the dataset small (~25k windows, ~300M
params): unfreezing more lets the network reshape everything to emit 88 numbers
and lose the structure that made the pretrained features useful. **Always keep
the no-fine-tuning frozen baseline as a control** — if it wins, fine-tuning is
hurting, and that is a real possible outcome here.

**`wav2vec2-large-960h` is deliberately not the default arm.** It is
`Wav2Vec2ForCTC` with `vocab_size=32` — ASR fine-tuned, so its top layers are
optimised to discard everything but which of 32 characters was spoken, which is
exactly the prosody signal we want. Use `facebook/wav2vec2-large-robust`: same
24 layers, self-supervised only, and the exact base audEERING fine-tuned from,
so it doubles as the control for the emotion arm.

**The emotion model is `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`.**
Confirmed against `../Clean_Irony/embeddings/`: its `audio_wav2vec_avd/*.npy`
are `(1, 3)` and `audio_wav2vec/*.npy` are `(1, 1024)` — both came from this one
model, whose forward returns *both* the 1024-d pooled state and the 3-d head.
- **It is pruned to 12 transformer layers, not 24.** Layer settings tuned for a
  24-layer backbone are invalid; `auto` resolves this per model.
- **Output order is arousal, dominance, valence** (`id2label` confirms), *not*
  arousal-valence-dominance. If any older irony analysis labelled those columns
  A/V/D in that order, columns 1 and 2 are swapped.

**Recommended arms.** A `--model wav2vec2-robust`, B `--model emotion`,
C `--model wav2vec2-robust --truncate-layers 12` (optional). A vs B answers
"which features predict better"; C vs B is needed to attribute a difference to
emotion pretraining rather than depth.

**Layer ranges shift after fine-tuning.** Registry `default_layers` are for
*base* models and read mid-network (`12-17`), because prosody peaks mid-stack.
After fine-tuning on prosody targets the upper layers are prosody-tuned, so
extract from `18-23` on fine-tuned 24-layer checkpoints. Layer choice is
empirical — sweep it with `--layers` and different `--out-name`s.

## Gotchas that cost real time

- **Fine-tuned checkpoints must be unwrapped.** `train_model` saves an
  `AudioEncoderForProsody` whose weights nest under `encoder.`.
  Loading that directory straight into `Wav2Vec2Model` silently drops them and
  substitutes random weights. `extract/wav2vec.py` detects and unwraps.
- **Feature/response TR alignment is checked, not assumed.** Features sit on a
  `respdict − TR_PAD` grid (verified: 261 → 256); stored responses may be on
  that grid or the raw one. `preprocess.trim_response` anchors both at their
  common end, logs the offset it found, and raises otherwise. Still unconfirmed
  on real data because the `.hf5` responses are OneDrive-dehydrated —
  **check the logged offset on the first real encoding run.**
- **The fine-tuning half of that alignment is confirmed** (2026-08-19): across
  all 83 split stories, `tr_onsets(story)[TR_PAD + 5 : ...]` matches the target
  `n_TRs` exactly — 83 aligned, 0 mismatches, 0 missing TR timing. A wrong
  `--trim` now raises in `finetune/dataset.py` instead of skipping the story
  with a print.
- **OneDrive on-demand files** raise `OSError: [Errno 5]`. Mark `data/` "always
  keep on this device" or point `FMRI_DIR` at local storage before long runs.
  Confirmed still dehydrated on 2026-08-19: `data/stimuli_16k/*.wav` have sizes
  but no content (`du` reports 0, reads give EIO, soundfile says "Format not
  recognised"). **Rehydrate before the first fine-tuning run** — nothing that
  touches audio can run until then.
- **Validation scalers come from training** (`get_fitted_scalers()`); refitting
  leaks and inflates metrics.
- `--min-ev 0.1` restricts fitting to voxels with real signal and cuts runtime a
  lot; results are scattered back to full voxel space.

## State as of 2026-08-19

- Code complete and import/lint clean; every CLI verified with `--help`.
- Encoding core, permutation test and contrast maps validated on synthetic data
  with known ground truth (delta ≈ 0 for unimodal voxels, strongly positive for
  voxels driven by both).
- The emotion checkpoint was verified to load with real pretrained weights,
  byte-identical through `AutoModel` and the fine-tuning wrapper.
- **Nothing has been run on real data yet.** No git commit has been made.
- **Fine-tuning is otherwise unblocked**: all 84 target JSONs already exist under
  `data/features/prosody/finetune_targets/averaged/` as `<story>_prosody.json`.
  Renamed 2026-08-20 from `brain_targets_finetuning/*_prosody+brain-pca-avg.json`
  and stripped of the dead `brain_targets` block; the audio features are
  bit-identical, and the originals are in `trash/brain_pca_multitask/`.
  Split is 71 train / 12 val / `wheretheressmoke` held out; 23,853 train and
  3,944 val windows. Only the dehydrated wavs stand in the way.
- The fine-tuning stack was reviewed and fixed on 2026-08-19: `--model <hf id>`
  crashed after dataset build; `--seed` did not reach model init; resume could
  only add freezing, never remove it; `bf16` was set without checking hardware
  support; `run_name` collided across `--llrd`/`--learning-rate`/`--seed`; a
  wrong `--trim` dropped stories silently. `--metric-for-best` was added
  (`eval_loss` is minimised by predicting each feature's mean; `eval_mean_r`
  is not). Verified with an end-to-end `train_model` run and a checkpoint
  unwrap round-trip.
- Gradient checkpointing was checked and is fine — but `freeze_base_model` must
  keep using its `named_modules` loop, **not** `encoder.freeze_feature_encoder()`.
  The HF helper clears `Wav2Vec2FeatureEncoder._requires_grad`, the flag whose
  `forward` uses to force `hidden_states.requires_grad`; clearing it severs the
  graph across the frozen bottom and the trainable layers get no gradient.
- Expect ~10 of the 88 targets to sit near the floor: every backbone sets
  `do_normalize: true`, so each 2 s window is z-scored and absolute level is
  gone from the input, while openSMILE computes `equivalentSoundLevel_dBp` and
  the loudness percentiles from the original signal. Not a bug.
- Data was **moved** here out of `../NL_Project`, which is now code-only and
  whose scripts will fail on missing data. That was intentional.

## Conventions

- `config.py` is the only place paths live; every one is env-overridable.
- Features: one `<story>.hf5` per story, dataset `data`, shape `(n_TRs, n_dim)`.
- Run with `python -m <package>.<module>` from the project root.
