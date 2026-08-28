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
- **Feature/response TR alignment — RESOLVED 2026-08-24.** The logged offset on
  the first real encoding run is **−15**, consistently across stories, and the
  stored responses are on a *third* grid the code did not know about: already
  trimmed to the final grid before storage. `adollshouse` is
  `respdict 261 → 256 feature TRs → 241 stored response TRs`, and
  `256 − TR_PAD − 2·trim == 241` exactly with `trim=5`.
  `preprocess.trim_response` accepted only offsets `0` and `TR_PAD`, so it
  raised on every story; it now recognises `offset == −(TR_PAD + 2·trim)` and
  returns such a response untouched. The near-miss matters more than the crash:
  cutting an already-cut response would drop 15 further TRs and shift responses
  against features by 10 TRs (20 s) while leaving the shapes plausible.
  Verified end-to-end on UTS01 — features and responses land on identical TR
  counts for every story, responses are NaN-free and already z-scored.
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
- **Stage 1 is done on real data (2026-08-24).** Two arms fine-tuned on the
  cluster and pulled back to `results/finetune/`:
  `wav2vec2_robust_frozen_12_lr3e-05_seed42` (best epoch 23, `eval_mean_r`
  0.6622) and `emotion_frozen_6_lr3e-05_seed42` (best epoch 29, 0.6534). Both
  verified to load with real fine-tuned weights: frozen bottom half is
  byte-identical to the pretrained base, trained top half differs by 3–10%.
- **Five audio feature bands extracted**, 84 stories each, identical TR grids
  (29,348 TRs): `opensmile` (88d), `base_robust_12to17`, `ft_robust_18to23`,
  `base_emotion_6to11`, `ft_emotion_6to11` (1024d each). Note the robust pair
  is extracted at *different* depths, so a base-vs-ft difference there is
  confounded with layer range; the emotion pair is matched at 6-11.
- **Encoding smoke test passed** (UTS01, 6 stories, banded/holdout, 7.2 min).
- **Only UTS01 has response data locally**; `SUBJECTS` lists nine. The other
  eight are ~19 GB each and are not on this machine in any form.
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

## Stage-2 result: the layer sweep is done (2026-08-28)

Full prosodic layer sweep run on the cluster: 9 subjects x 4 stores = 36 GPU
tasks, 40 training stories each, 5-fold nested CV, `--min-ev 0.1`, held-out
story never touched. All 36 complete. Summarise with
`python scripts/summarise_sweep.py`.

**Fine-tuning on eGeMAPS hurt, and the damage scales with how far each layer
moved.** Emotion model, paired per subject at matched depths:

    layer   base      ft      ft-base   subjects worse
    L6      0.0192    0.0191  -0.0000   5/9   <- freeze boundary, same weights
    L9      0.0200    0.0186  -0.0015   8/9
    L11     0.0204    0.0151  -0.0053   8/9

That mirrors the weight divergence measured before any brain data was involved
(corr(ft, base) +0.91 at L6, +0.12 at L11). The frozen base *rises* to its top
layer (0.0192 -> 0.0204); the fine-tuned one falls away. The control predicted
in the design notes won, so **carry the frozen base forward, not the fine-tuned
checkpoint**: `base_emotion` L9-L11 and `base_robust` L15-L18.

The robust pair is a much weaker case (±0.001, inconsistent sign) — its base
top layers were already degraded, so there was less to spoil. Do not
over-generalise from it.

Every learned layer beats openSMILE, by +0.0017 to +0.0059. Best mean is
`base_emotion` L11; most consistent is `base_robust` L18 (9/9 subjects).
Note `base_emotion` is still climbing at its final layer — the stack ran out
before the profile did.

Scale: mean r ~0.02 over the 1,776 of 81,126 voxels that pass EV>0.1; best
voxel ~0.09. Real and consistent, but differences between small numbers.

Published summary with the profile charts:
https://claude.ai/code/artifact/1b2b34ce-2eb3-447f-9809-35d5bbd4f39d

## Solver settings that cost real time (2026-08-28)

Measured on an A100 80GB at the sweep's shapes, n=13,329, p=352, 1,776 targets,
5 folds (`scripts/bench_solvers.py`); **all variants score identically**, so
these are pure cost:

    eigh                     9.1 s
    svd                    385.8 s      <- what default_solver_params used
    GroupRidgeCV (primal)    1.0 s
    RidgeCV (primal, svd)    0.6 s

- **`diagonalize_method="svd"` was costing 42x.** Now `eigh`, with an automatic
  per-fit fallback to `svd`, because eigh genuinely does fail on some subjects:
  a linear kernel from p features has rank <= p, so p=4096 against n=9,461 TRs
  leaves ~5,000 zero eigenvalues and LAPACK will not converge. It hit the four
  subjects with 27 stories rather than 84.
- **`--n-splits` reached only the outer loop.** The inner CV called
  `story_folds()` with no `n_splits`, i.e. leave-one-story-out, so a 40-story
  sweep ran 5 x 32 = 160 fits per configuration instead of 25. Fixed via
  `fit_banded_cv(inner_n_splits=...)`; `run_encoding` still leaves it unbounded,
  which is right for a final single-configuration model.
- **Primal vs dual is not the win the Gallant tutorials imply here.** At p=4096
  the dual is *faster* than primal on GPU, because its n x n eigendecomposition
  is amortised over the whole alpha grid and is nearly p-independent. Only the
  88-d openSMILE band is in the regime where primal pays. But the dual is
  inherently rank-deficient at p < n, which is what forces the svd fallback —
  worth revisiting if that fallback starts firing on most fits.

## Running an array this wide: two traps

- **himalaya's progress bar kills array jobs.** It redraws stdout thousands of
  times per fit; under SLURM stdout is a file, and 36 concurrent tasks got
  `OSError: [Errno 121] Remote I/O error`. `default_solver_params` now passes
  `progress_bar=False`.
- **`/home` is a BeeGFS volume at 97% full and refuses log writes at that rate.**
  Symlink `logs/` to scratch before submitting. The failures are worse than
  cosmetic: tasks that had already written `sweep.csv` were marked FAILED
  because the closing `echo` could not reach the log, so "rerun the failures"
  redoes finished work. **Check the output files, not the SLURM exit states.**

## Open questions — noted 2026-08-28, not acted on

From a multimodal encoding pipeline the user read (the description matches
Meta's TRIBE / Algonauts 2025 setup): timed text embeddings from
**Llama-3.2-3B** with k=1024 words of preceding context, per-layer, summed into
2 Hz bins; audio from **Wav2Vec-BERT-2.0** over 60 s chunks, resampled 50 Hz to
2 Hz, per-layer, 1024d.

**Llama for the semantic band — yes, eventually.** GPT-2 small is a weak
language model by current standards, and the case for upgrading is not
speculative on *this* dataset: Antonello et al. 2023 ("Scaling laws for
language encoding models in fMRI") ran the LLaMA/OPT families on ds003020 and
found encoding performance scales with model quality well past GPT-2. Two
things to check before copying the recipe: the quoted `Dtext = 2048` is
Llama-3.2-**1B**'s hidden size (3B is 3072), so the paper's own numbers do not
line up with the model it names; and k=1024 words of context is far longer
than a GPT-2 window, so part of any gain is context length, not model size.
Consequence for us: a stronger semantic band makes `delta` and `preference`
*harder* for audio, which is the conservative direction — good science, but
every prosody number moves when it lands. This is exactly why the current sweep
excludes the semantic band entirely.

**Wav2Vec-BERT-2.0 for the audio band — no, not in these arms.** It is a
stronger speech encoder (SeamlessM4T's, 4.5M hours), but the whole point of
`wav2vec2-large-robust` here is that it is the exact base audEERING fine-tuned
from, so it doubles as the matched control for the emotion arm. Swapping the
backbone breaks that pairing and leaves "emotion pretraining vs better speech
model" confounded. Worth a separate arm once the current comparison is settled;
not a drop-in.

**Two details worth stealing now, both cheap and backbone-independent:**
- *Audio context window.* They feed 60 s chunks; we mean-pool a 2 s window per
  TR. The 2 s matches the eGeMAPS target windows, which was right for
  fine-tuning, but for *extraction* a longer window gives the transformer real
  context. Testable with the existing code.
- *Causal/bidirectional asymmetry.* Their note that audio embeddings see the
  future while text embeddings do not applies to us too: our audio band is a
  bidirectional transformer over its window, our GPT-2 band is causal. That is
  a genuine confound in `preference = r_text - r_audio` — the audio band gets
  information the semantic band is structurally denied. Worth stating in the
  paper at minimum, and worth a causal-masked control if a reviewer asks.


## Conventions

- `config.py` is the only place paths live; every one is env-overridable.
- Features: one `<story>.hf5` per story, dataset `data`, shape `(n_TRs, n_dim)`.
- Run with `python -m <package>.<module>` from the project root.
