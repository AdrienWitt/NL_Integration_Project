# The audio bands were misaligned by 5 TRs. Fixed and redone (2026-09-09)

`tr_onsets` returned the *scanner* clock and used it to index the wav, so every
audio window covered `[2i, 2i+2)` where it should have covered `[2i-10, 2i-8)`
— ten seconds, five TRs, later than the text band and the responses it was
regressed against. Fixed in `common/tr_alignment.py`. All five audio stores and
the eGeMAPS fine-tuning targets have been rebuilt and every prosody sweep re-run,
so **there is nothing left to redo**; what follows is why it mattered, because
the size of the correction is what dates several sections below.

Measured on UTS04-09 only — their repeat count did not change, so the EV mask is
identical and *only the audio moved*:

    store            old grid            new grid
    base_emotion     L10  0.0207         9-11  0.1520      x7.3
    ft_emotion       L9   0.0171         L7    0.1238      x7.2
    base_robust      L17  0.0177         L11   0.1526      x8.6
    ft_robust        L12  0.0172         L12   0.1485      x8.6

openSMILE per subject: 0.0222→0.1280, 0.0050→0.1038, 0.0106→0.0872,
0.0064→0.0816, 0.0070→0.0679, 0.0026→0.0866 — x5.8 to x33.9.

The previous version of this file called the prosody numbers "a floor, not an
estimate". The floor was an order of magnitude low, and the audio-vs-text
comparison this project exists to make is only now being run against a real
audio band.

**Not affected, and never were:** every text band (`DataSequence` uses
`get_reltriggertimes()` directly and was always on the sound clock), the
responses, and the two fine-tuned checkpoints — input windows and eGeMAPS
targets were shifted *together*, so the mapping they learned is intact. That
last claim is now measured rather than argued; see the next section.

A side benefit of the fix: the padded rows now fall at the *start* of each story,
where `[TR_PAD + trim : -trim]` removes them, instead of at the end where one or
two survived into the design. Confirmed on the rebuilt stores — `wheretheressmoke`
has 4 consecutive duplicate rows, all at the head and none at the tail; the old
store had the same 4 at the tail.

## The fine-tuning targets: rebuilt, and guarded so this cannot recur

The eGeMAPS targets under `data/features/prosody/finetune_targets/averaged/` came
from the same broken `tr_onsets`. Both `prep/make_finetune_targets.py` and
`finetune/dataset.py` call it, which is precisely why the checkpoints survived:
window and label were wrong together, so the learned mapping — audio window to
the acoustics of that same window — is unchanged.

Rebuilt 2026-09-09 (`scripts/cluster/make_targets.sbatch`, 44 min, CPU only) and checked
against both openSMILE stores on the same `[TR_PAD+trim : -trim]` slice:

    targets              vs new store    vs old store
    rebuilt                1.0000          0.05-0.16
    old (2026-08-20)       0.04-0.11       0.52-0.64

**1.0000, not 0.99.** The fine-tuning target and the openSMILE encoding band are
now the same numbers on the same grid, which matters for interpretation and not
only for plumbing: "the fine-tuned representation converges toward openSMILE's
encoding score" becomes a statement about one feature set rather than two
similar ones.

Two things changed with the rebuild:

- **`--audio-dir data/stimuli_16k` is required.** The script defaults to
  `STIMULI_DIR`, the native-rate wavs, which do not exist on the cluster. It is
  also the more correct choice: the model is fed 16 kHz, so functionals
  describing content above 8 kHz cannot be predicted from what it hears. That is
  what capped the old targets at r=0.56 against the store *on their own grid*.
- **The grid is recorded and verified.** Each target JSON now carries
  `first_onset_sec`, `audio_sampling_rate` and `audio_dir`, and
  `ProsodyDataset._check_grid` refuses to build when the recorded onset
  disagrees with what `tr_onsets` returns at run time — naming the gap in
  seconds and in TRs. A file lacking the field is refused too, since every JSON
  written before 2026-09-09 is on the old clock. Without this a fine-tuning run
  launched after the fix would have paired audio from `[2i-10, 2i-8)` with
  labels from `[2i, 2i+2)` on every window, silently, with a normal-looking loss
  curve. The old targets are kept at `finetune_targets/_oldgrid_20260909/`.

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
"which features predict better"; C vs B was there to attribute a difference to
emotion pretraining rather than depth.
**A vs B is now answered, and it is a tie** (+0.0004, 4/9 — see the corrected
Stage-2 section), so C is no longer worth running: there is no difference to
attribute. A is the simpler arm to report.

**Layer ranges: sweep them, and ignore the old advice about fine-tuned
checkpoints.** This used to say "extract from `18-23` on fine-tuned 24-layer
checkpoints" because fine-tuning re-tunes the upper layers toward prosody. The
corrected sweep says the opposite: `ft_robust` is *best at L12* and falls
monotonically to L23, and `18-23` is among the worst ranges available. Prosody
peaks mid-stack in the base models (L11 of 24, L11 of 12) and fine-tuning does
not move the peak upward — it flattens everything it touches toward openSMILE.
Layer choice is empirical; sweep it with `--layers` and different `--out-name`s.

## Gotchas that cost real time

- **Check what already exists before computing anything.** This file lags the
  cluster: runs finish on Baobab before they are written up here (the Llama
  `wordrate` fit and null, 13417892/13417894, were done a day before this file
  still called the test open). Before submitting or launching, inventory both
  sides and compute only what is missing:
  Baobab `squeue -u $USER`, `sacct -S <recent> -X`,
  `ls -dt results/encoding/*/ | head`, and count `*/meta.json` / `*/report.json`;
  locally `ls -lt results/group/ /root/prosody_nulls/` and the exact output dir.
  Never rerun a model that has already run.

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
- **The two per-layer store conventions differ by one.** `extract.wav2vec
  --per-layer` writes index *i* meaning transformer block *i*;
  `extract.context_lm --per-layer` writes `hidden_states` indices, so 0 is the
  *embedding* layer and block *i* is at *i + 1* — the same numbering
  `--layers` already took on the text side. Both label the axis in the `layers`
  attribute; read that rather than assuming. A 13-entry GPT-2 store is 12
  blocks plus embeddings, not 13 blocks.
- **`scripts/make_common_stories.py` needs `PYTHONPATH=.`** — it is a script,
  not a package module, so `python scripts/make_common_stories.py` alone dies
  on `No module named 'config'`.

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
- **Five audio feature bands extracted**, 84 stories each. SUPERSEDED: these
  flat stores were on the misaligned grid and the per-layer stores replaced
  them. What exists now is `opensmile` (88d) plus `perlayer_base_robust` (0-23),
  `perlayer_ft_robust` (12-23), `perlayer_base_emotion` (0-11) and
  `perlayer_ft_emotion` (6-11), all 1024d and all on the corrected grid. The
  per-layer form also removes the old confound where the robust base/ft pair was
  extracted at different depths.
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

## Stage-2 result: every layer, common stories (2026-09-09, corrected grid)

Supersedes both earlier prosody sweeps. The 2026-08-29 version ran on the
misaligned audio and is wrong about the peak layer, the direction of the
fine-tuning effect, and the value of emotion pretraining — three of its four
conclusions. Archived at `results/encoding/prosody_sweep_misaligned_20260909/`.

Design unchanged: 9 subjects x 4 stores = 36 GPU tasks, all nine on the same 24
training stories (`common_stories_all9.json`), every stored layer plus averaged
ranges, `--min-ev 0.1`, `--max-repeats 5`, both `--eval cv` and `--eval holdout`.
Summarise with `python scripts/summarise_sweep.py --eval cv --tidy out.csv`.

**Choose on cv, report on holdout** — and the cost of getting that backwards is
now measured rather than asserted. Over the 36 (store, subject) sweeps, the cv
argmax and the holdout argmax agreed in **1 of 36**, disagreeing by 3.0 layers on
average; selecting on holdout would have inflated the reported score by **+0.0083
on average** (median +0.0067, max +0.027) — most of the entire effect. The
held-out story is 291 TRs and its standard errors run about 4x the
cross-validated ones.

Mean r over the EV>0.1 voxels, averaged over the nine subjects:

    store                  cv      vs oSMILE   n      holdout
    opensmile            0.0867        —               0.1127
    base_emotion  9-11   0.1516     +0.0649   9/9      0.2963   <- best on cv
    base_robust   L11    0.1512     +0.0646   9/9      0.2796
    ft_robust     L12    0.1463     +0.0596   9/9      0.2749
    ft_emotion    L7     0.1190     +0.0323   9/9      0.2225

**`base_robust` peaks at L11 of 24, a clean inverted U** — not the "cliff at
L19-20" the misaligned sweep reported:

    L0 .099  L2 .107  L4 .113  L6 .122  L8 .136  L10 .151  L11 .152  <- peak
    L12 .147 L14 .135 L16 .127 L18 .118 L20 .100 L22 .099 L23 .103

That also settles the truncation worry about the emotion model. It is
`wav2vec2-large-robust` pruned to 12 layers, its own profile rises to L11, and
the 24-layer version shows the peak is real and then falls.

**Emotion pretraining buys nothing.** `base_emotion 9-11 - base_robust L11 =
+0.0004`, emotion ahead in 4/9. Before the fix emotion was the clear winner and
the band to carry forward; it is now a tie, with the cheaper, unpruned,
self-supervised backbone reaching the same place. Arm C (`--truncate-layers 12`)
is no longer needed to attribute a difference, because there is no difference to
attribute.

**Averaged ranges still do not pay.** emotion 9-11 (0.1516) vs L11 (0.1504):
+0.0012. This used to say "for 3x the columns", which was wrong: a range
*averages* its layers (`run_prosody_sweep.py:156`, and `common/io.py` for the
`store:9-11` syntax), so it is the same 1024 dimensions. The cost is three
stored layers instead of one, not three times the design matrix.

**DECIDED 2026-09-10: carry `perlayer_base_emotion:11`.** The three candidates
sit within 0.0012 of each other, so this is a choice of story, not of
performance, and the story is what settles it: the affective band
(`emotion_avd`, see below) is the *head* of this same checkpoint, so 3-d against
1024-d is an affective bottleneck measured against the representation it is
drawn from, rather than a contrast across two backbones. The single layer over
9-11 for the reason above — same dimensionality, one stored layer.
`base_robust:11` remains the fallback if the emotion-pretraining caveat ever
needs removing; it is 0.0008 away.

## Why fine-tuning on eGeMAPS hurts, measured (2026-09-09)

Fine-tuning degrades brain prediction in **9/9 subjects**, and the shape of the
damage says why. ft minus base at matched depth, cv:

    ft_emotion (0-5 frozen, 9 subj)        ft_robust (0-11 frozen, 7 subj)
      L6  .1215 -> .1176  -0.0039  9/9       L12 .1473 -> .1474  +0.0002  1/7
      L7  .1314 -> .1190  -0.0125  9/9       L15 .1300 -> .1300  -0.0000  5/7
      L8  .1422 -> .1165  -0.0257  9/9       L18 .1180 -> .1167  -0.0013  7/7
      L9  .1492 -> .1094  -0.0398  9/9       L20 .0996 -> .1043  +0.0047  0/7
      L10 .1494 -> .1031  -0.0463  9/9       L21 .0995 -> .1019  +0.0024  1/7
      L11 .1504 -> .0940  -0.0564  9/9       L23 .1031 -> .1006  -0.0025  7/7
    openSMILE alone: 0.0867                openSMILE alone: 0.0894

Three facts fix the interpretation:

1. the damage grows **monotonically with depth into the trainable region**;
2. it **stops at openSMILE's own score** (ft L11 = 0.0940 against 0.0867);
3. where a base layer scored *below* openSMILE, the same objective made it
   **better** (robust L20-22, 0/7 worse).

So the loss transports every trainable layer toward one destination — a
representation sufficient for 88 numbers — reached from above or from below, and
the only free variable is how far along that path training travels. eGeMAPS is a
weaker band than the pretrained mid-stack representation, so distilling into it
loses wherever the layer was already stronger.

This is also why `robust` looks unharmed: **its fine-tune froze the peak.** It
trains blocks 12-23 while the peak sits at L11, so it never touched the layer
that matters. The emotion fine-tune froze 0-5 and trained 6-11, which is exactly
where the information is. Not one backbone being sturdier than the other — one of
them fine-tuned where there was nothing left to lose.

The registry's warning was right, and the frozen-baseline control it insisted on
is what caught it: "if it wins, fine-tuning is hurting, and that is a real
possible outcome here."

### CLOSED 2026-09-10: there is no drift worth testing

A ridge from a **frozen**, mean-pooled layer to the same 88 functionals, on the
same split (23,853 train / 3,944 val windows) and the same per-column r averaged
over 88 targets — so directly comparable to `eval_mean_r`:

    base_robust  L6   0.6359      base_robust L20  0.6282
    base_robust  L11  0.6184      ft_robust   L12  0.6231
    base_emotion L11  0.6145      base_robust L12  0.6174
    full fine-tuning, 32 epochs   0.6622

Fine-tuning buys **+0.026 on its own target** for up to **-0.056** on the brain.
And the eGeMAPS-decodability profile is nearly flat across depth (range 0.018)
while brain prediction is a sharp inverted U (range 0.052): **eGeMAPS content is
not what makes L11 the best brain-predicting layer**, so moving a layer toward
eGeMAPS cannot add anything there — only remove. The destination contains
nothing the backbone did not already have linearly, which is why there is no
non-monotonic optimum to find and no lambda sweep worth paying for.

**The rule this leaves behind, and it generalises past eGeMAPS: a target
deserves fine-tuning only if a linear readout of the frozen backbone cannot
already reach it.** Run the probe first. Both candidate targets here fail it for
opposite reasons — eGeMAPS is already linearly present, and arousal/dominance/
valence was already learned by audEERING on far more data than this dataset
holds.

Keep the negative result; it is publishable as written. The levers below stay in
the code, unused, in case a future target passes the probe.

**Two levers exist to test whether a *small* drift helps before a large one
destroys.** The mechanism predicted a non-monotonic optimum, and nothing in the
old setup could express "let L11 move a little" — freezing is binary per layer.
Both default to off, so an unflagged run reproduces the existing checkpoints and
keeps its directory name.

- `--l2sp LAMBDA` — weight decay toward the *pretrained* weights instead of zero
  (Xuhong et al. 2018), penalty `0.5*LAMBDA*||theta - theta_0||^2`. Anchors are
  taken after freezing and held as non-persistent buffers, so they follow the
  model onto the GPU without doubling every checkpoint on disk.
- `--pool-layers weighted` — the head reads a learned softmax over every hidden
  state rather than `last_hidden_state`, so the gradient stops concentrating on
  the layer feature extraction reads. The learned weights are themselves a
  result: they say which depth the eGeMAPS objective actually wanted.
- `DriftCallback` writes `metrics/drift.json` — per-layer
  `||theta - theta_0|| / ||theta_0||` at every evaluation, plus the pooling
  profile. LAMBDA is in units nobody has intuitions about; drift is not.

~~**The cheapest experiment is one run with a large `--save-total-limit`**~~ —
superseded by the frozen probe above, which answered the same question for the
cost of a CPU job: the optimum is at zero drift, because the destination holds
nothing new. `scripts/cluster/finetune.sbatch` takes `SAVE_LIMIT` as an env var if a
future target ever justifies the curve.

Note `--llrd` already exists in `finetune/optim.py` and has never been used: both
published runs are `frozen_N_lr3e-05_seed42`. It is the softer version of the
freeze boundary and costs nothing to try.

## The affective band: `emotion_avd`, and what it can and cannot claim

Submitted 2026-09-10 (Baobab 12590825 -> 12590826). `extract/emotion_avd.py` has
existed since August and had never been run.

Same checkpoint the encoding already uses, read at the **head** instead of the
hidden states: its forward returns both the 1024-d mean-pooled state and the 3
numbers the regression head makes from it. The audio band is otherwise 1024
opaque dimensions, so "this voxel is predicted by the audio band" says nothing
about *what* it tracks. Comparing r(3-d) with r(1024-d) asks how much of the
audio effect survives an affect-only bottleneck — the only measurement here that
addresses *affective* prosody rather than prosody.

Measured 2026-09-10, all nine subjects, one runner, one design, `common_
stories_all9`, `--min-ev 0.1`, `--max-repeats 5`, each band a single-band
banded ridge so the rows differ only in the band:

    band                 dim    columns      cv     holdout
    emotion_avd            3        12    0.0573    0.0911
    opensmile             88       352    0.0866    0.1131
    base_emotion L11    1024      4096    0.1505    0.2908

    r_avd / r_emotion            cv 0.379   (per subject 0.33-0.45)
                            holdout 0.305   (per subject 0.17-0.43)
    avd < opensmile              9/9 subjects on cv, 8/9 on holdout

**Three named affective dimensions recover ~38% of what 1024 learned ones
explain**, and the fraction is stable across subjects. That is the quantity
this project existed to produce, and it is substantial for three columns.

**But affect alone sits below openSMILE in 9/9 subjects.** The ordering is
affect (3d) < hand-designed acoustics (88d) < learned representation (1024d).
So affective prosody is an identifiable and sizeable *component* of what the
audio band explains, not its principal explanation — write the paper that way.
Two caveats belong in the same sentence: these are A/D/V *predicted from
acoustics*, and since dominance ~ arousal the three dimensions are effectively
two.

A cross-check worth keeping: the sweep reported **0.1504** for this band and
`run_encoding` gives **0.1505**, through two different runners and two
different alpha searches.

Do **not** use `--output both`. The 3 values are a function of the 1024-d
vector, so concatenating returns an opaque band and throws away the only thing
worth having: three columns with names.

**Dominance is not a separate dimension.** Measured on the LeBel stimuli
themselves (84 stories, 28,088 TRs on the design slice), and on the 774 stimuli
of `../Clean_Irony/embeddings/audio_wav2vec_avd/` for comparison:

                          LeBel     Clean_Irony
    arousal x dominance   0.963       0.950
    arousal x valence     0.416       0.223
    dominance x valence   0.408       0.274
    variance/component    0.734       0.639
                          0.254       0.348
                          0.011       0.013

**Effective rank 2 on both**, so the dominance result is a property of the model
(MSP-Podcast dominance annotations track arousal), not of a corpus — and on
LeBel it is slightly worse, 93% shared variance. Consequences:

- **Valence vs arousal is the identifiable contrast**, and the one with a
  literature behind it. But it is *less* clean here than the irony corpus
  suggested — r = 0.42 rather than 0.22, and the per-story median is +0.401 with
  a range of [-0.007, +0.674], so how separable the two are depends heavily on
  the story. Report it as primary, and put the cross-subject replication gate on
  it too rather than only on dominance.
- **Arousal vs dominance is not.** Banded ridge does not dissolve collinearity,
  it *allocates* it: with 90% shared variance the split is decided by noise in
  the model's own output, and a winner-take-all map still looks clean because an
  argmax never abstains. Gate any per-dimension claim on cross-subject
  replication — per voxel, how many of the nine subjects agree, against
  Binomial(9, 0.5); the exact sign test bottoms out at 1/512 = 0.002, the same
  floor as every other group claim here.
- Whether the third dimension earns its columns is `--models arousal+valence`
  against `joint`: `r_joint - r_(arousal+valence)` is the unique contribution of
  dominance, the same conditional-contribution logic as the text/audio
  permutations, one level down. This is deliberately *not* a comparison against
  openSMILE or the emotion layer, which answer "which band predicts better".

Read the run back with `PYTHONPATH=. python3 scripts/avd_contrast.py`.

### Which dimension earns its columns: measured (2026-09-11, all seven models)

Seven models rather than five (`cf8b0dd`), so every dimension gets the same
leave-one-out treatment dominance already had. `r_joint - r_subset`, nine
subjects, the same design as the table above:

    dimension       cv      n    p        holdout     n    p
    arousal      +0.0187   9/9  0.002     -0.0110    1/9  0.998
    dominance    +0.0227   9/9  0.002     -0.0109    1/9  0.998
    valence      +0.0041   8/9  0.020     +0.0222    9/9  0.002

**Only valence is positive on both evals, and the two evals otherwise say
opposite things.** On cv all three dimensions add something and dominance adds
the most; on holdout arousal and dominance both come out *negative* in 8 of 9
subjects, i.e. the joint model (0.0872) scores below both `arousal+valence`
(0.0981) and `dominance+valence` (0.0982). A nested joint model scoring below
its own subset is noise, not a finding: 291 TRs and a randomised band-weight
search. But the direction of that noise is not arbitrary — it is arousal and
dominance, the pair at r = 0.963, whose split is allocated rather than measured,
and the collinear pair is exactly where a re-draw of the simplex can move the
answer.

**So report valence.** It is the one dimension whose unique contribution is
positive under both evaluations and in 9/9 subjects on the one that was never
used for selection. Dominance's own sbatch header already recorded the sign
flip; the arousal row now shows the flip is a property of the collinear *pair*,
not a fact about dominance. Do not report a per-dimension map for arousal or
dominance without the cross-subject replication gate above.

The one thing both evals agree on beyond valence: the three-dimension joint is
not bigger than a well-chosen two-dimension subset. `arousal+valence` and
`dominance+valence` score within 0.0001 of each other on holdout, which is what
an effective rank of 2 looks like when you ask it directly.

**`run_encoding` now takes N named bands** (`6f8e28d`): `--band NAME=STORE`,
repeatable, replacing the text/audio pair, because a decomposition *inside* one
modality is three bands of one modality rather than a text band and an audio
band. Each band still gets its own alpha; `split_corrs` still reports each
band's share of the joint prediction — shares that sum to the joint r, **not**
standalone correlations. Without `--band` nothing changes: the two-band design
matrix is column-for-column identical, and `MODEL_BANDS` keeps its name because
`stats.run_permutation` imports it.

**Do not raise `--n-iter` here.** Set to 200 on the reasoning that 12 columns
makes each fit cheap, it timed out all nine tasks at 4 hours. `random_search`
samples the simplex of *band weights*, so its cost scales with the number of
bands, not columns, and the fit count is `outer * inner * n_iter`. Measured on
UTS01 at n_iter=200: 1 min 45 for each single-band model (the simplex is a
point) and **over 3 h 54 for the two-band model, unfinished**. 5 x 23 x 200 =
23,000 fits per model against a sweep's 500.

**The inner CV size does not matter, measured** (`scripts/cluster/inner_cv_check.sbatch`,
2026-09-10). `base_emotion:11` on cv, alpha search leave-one-story-out against
5-fold, three subjects spanning mask size:

    UTS02  0.1493 / 0.1492      UTS04  0.1770 / 0.1770
    UTS07  0.1534 / 0.1533      mean |diff| 0.00007, max 0.0001

An order of magnitude below anything reportable, so the sweeps' bounded search
was the right call -- 4.6x cheaper for 0.0001 -- and their layer choices are
safe under leave-one-story-out. Note both settings are nested (the search runs
on `story_ids[train_idx]` only), so this was never a leakage question; the one
place leakage is real remains `--backend huth --eval cv`.

`stats/run_permutation.py` is still text/audio only, so the per-dimension claims
carry counts rather than p-values until it is extended.

## Stage-2 result: semantic context length (2026-09-01)

> **On the old EV masks.** Predates `--max-repeats 5`, so UTS01-03 are
> scored over 10-repeat masks. The k=16 *choice* is safe — the mask is the
> same for every candidate within a subject — but the numbers are not
> comparable to the prosody table above. See "The final joint model".

9 subjects x 7 configurations x both evals, all nine on `common_stories_all9`,
`--min-ev 0.1`, GPT-2 small at `--layers last`. Complete; read with
`python scripts/summarise_sweep.py --root results/encoding/semantic_sweep/cv
--baseline gpt2_mean`.

Mean r over the EV>0.1 voxels, averaged over the nine subjects:

    config       cv       Δ        holdout   Δ
    gpt2_mean   0.1346    —        0.2683    —
    gpt2_k0     0.1417   +0.0070   0.2823   +0.0140
    gpt2_k1     0.1500   +0.0153   0.2995   +0.0312
    gpt2_k4     0.1596   +0.0249   0.3150   +0.0468
    gpt2_k16    0.1657   +0.0310   0.3271   +0.0588   <- best on cv
    gpt2_k64    0.1587   +0.0241   0.3233   +0.0550
    gpt2_k256   0.1654   +0.0308   0.3312   +0.0629

**Context helps in 9/9 subjects**, +0.031 cv. This used to say it was the
biggest effect in the project, against the ~+0.010 the best audio layer bought
over openSMILE — that comparison was against the misaligned audio band. On the
corrected grid the best audio layer buys **+0.065** over openSMILE, so the
audio-layer effect is roughly twice the context effect, not a third of it.
Context still helps in 9/9; it is no longer the headline.
k=16 and k=256 are indistinguishable; the dip at k=64 is noise.
Choosing on cv: **k=16**, same score for a sixteenth of the context.

> **SUPERSEDED 2026-09-15 — this whole grid ran at L12.** "k=16 and k=256 are
> indistinguishable" is true at L12 and false at L8, where k=256 wins by
> +0.0154 in 9/9 subjects. The k choice made here was measured at the one layer
> that hides the interaction, exactly as the depth section below was measured at
> the one layer least likely to win. See "k at L8, CLOSED" for the band that
> replaces it.

## Stage-2 result: GPT-2 depth at k=16 (2026-09-08, re-scored 2026-09-11)

> **Re-scored on the current `--max-repeats 5` masks and it held.** The table
> below is the new one; the old-mask run is archived at
> `results/encoding/semantic_sweep_oldmask_20260909/`. Everything the old
> version claimed survived — same peak, same shape, same +0.0397 against
> `gpt2_mean` to four decimals — so the two paragraphs of caveat that used to
> sit here are gone rather than updated.

The context sweep above ran entirely at `--layers last`, which on a *causal*
LM is the layer least likely to win: the top of the stack is optimised to emit
next-token logits and rotates back toward the unembedding matrix, away from
the abstract middle (logit lens, nostalgebraist 2020; tuned lens, Belrose et
al. 2023). So +0.031 was measured at a probable trough. It was.

9 subjects x 13 hidden states x both evals, k=16, `common_stories_all9`,
`--min-ev 0.1`. Mean r over the EV>0.1 voxels, averaged over subjects:

    layer     cv      vs L12   n     holdout   vs L12   n
    L0      0.1347   -0.0188  0/9    0.2584   -0.0401  0/9
    L1      0.1394   -0.0141  0/9    0.2699   -0.0286  0/9
    L2      0.1423   -0.0112  0/9    0.2731   -0.0255  0/9
    L3      0.1458   -0.0077  0/9    0.2792   -0.0194  2/9
    L4      0.1493   -0.0042  0/9    0.2871   -0.0115  2/9
    L5      0.1529   -0.0006  4/9    0.2929   -0.0057  4/9
    L6      0.1565   +0.0030  8/9    0.2981   -0.0004  5/9
    L7      0.1599   +0.0064  9/9    0.3051   +0.0065  7/9
    L8      0.1615   +0.0080  9/9    0.3101   +0.0116  8/9   <- peak, both evals
    L9      0.1607   +0.0072  9/9    0.3075   +0.0089  8/9
    L10     0.1588   +0.0053  9/9    0.3077   +0.0091  9/9
    L11     0.1559   +0.0024  9/9    0.3044   +0.0058  9/9
    L12     0.1535    —       0/9    0.2985    —       0/9   <- what the k sweep used

**A clean inverted U peaking at L8 of 12 — two thirds of the way up.** The
hourglass, measured. `--layers last` cost +0.0080 mean r. (This used to add
"about what the entire audio-layer effect buys over openSMILE (~+0.010)" — that
was the misaligned audio band; the corrected figure is +0.065, so depth on the
text side is an eighth of it, not a match for it.)

**L8 is the argmax in eight of nine subjects independently** on cv, UTS03 at L9
— not a group mean with a soft peak, eight separate maxima at the same layer.
(The old-mask run had 9/9; the one defection is the price of UTS03's mask
growing from 10-repeat to 5-repeat EV, and it is one layer away.) Holdout puts
the peak at L8 in five subjects, L10 in two, L9 and L11 in one each — the
*identical* distribution to the old masks, the same shape resolved less
sharply, exactly as its ~4x larger standard errors predict. Read L8-L10 as a
plateau there, and take the choice from cv as the rule says.

**The mask change moved every absolute number and no conclusion.** Mean r falls
about 0.013 on cv and 0.029 on holdout across the board — UTS01-03 are now
scored over 3.7x more voxels, which are on average less reliable — but the
profile is rigid: L8 peak on both evals, +0.0397 against `gpt2_mean` on cv to
four decimals on both mask sets, and the `gpt2_k16` / `perlayer_gpt2_k16:12`
identity still exact.

**The peak is not near the bottom**, which settles the interpretive worry that
motivated the sweep: at two thirds depth the band is not a lookup table, and
calling it semantic in `preference = r_text − r_audio` is defensible. Report
the profile alongside the argmax anyway — the shape is the evidence.

**Carry `perlayer_gpt2_k16:8` forward as the semantic band.** Against the
original `gpt2_mean` it is +0.0397 cv (context +0.0317, depth another +0.0080),
9/9 subjects.

Consistency check, built into the run: `gpt2_k16` (the flat store the context
sweep used) and `perlayer_gpt2_k16:12` are the same layer from the same
forward passes, and their encoding scores agree to **0.00e+00 in every
subject**. The per-layer path is verified end to end, not just at the feature
level.

Read it with:

    python scripts/summarise_sweep.py --root results/encoding/semantic_sweep/cv \
        --baseline gpt2_mean

## CLOSED 2026-09-15: k at L8 — depth and context do interact

They interact, the effect is large, and it moves the semantic band. The k grid
was run at L12 and the depth grid at k=16, so the one cell neither covered is
the one that matters. Three context lengths, every layer, nine subjects,
`common_stories_all9`, `--min-ev 0.1 --max-repeats 5`. Mean r over the EV>0.1
voxels, averaged over subjects, cv:

    layer      k16      k256      k512
    L0      0.1347    0.1526    0.1508
    L4      0.1493    0.1666    0.1657
    L6      0.1565    0.1730    0.1718
    L7      0.1599    0.1766    0.1764
    L8      0.1615    0.1769    0.1770   <- argmax at every k, both evals
    L9      0.1607    0.1746    0.1749
    L10     0.1588    0.1707    0.1707
    L12     0.1535    0.1530    0.1541   <- where the k grid was run

**Read the bottom row against the middle one.** At L12 the three context
lengths are within 0.0011 of each other, which is what the context sweep saw
and why it chose k=16 — "same score for a sixteenth of the context". At L8 the
same comparison is +0.0154 for k=256 over k=16, in **9/9 subjects** on cv and
9/9 on holdout. The context effect is not small; it was measured at the layer
where it is invisible.

**Depth and context interact in magnitude, not in location.** L8 is the argmax
at k=16, k=256 and k=512, on both evals — so the hourglass argument survives
intact and only the height of the peak moves. That is the cheap outcome: had
the argmax moved with k, every layer choice in this file would have needed
redoing per context length.

**Context saturates between 256 and 512, so k=256 is a ceiling and not an
edge.** k512:8 - k256:8 is **+0.0001, 5/9 subjects** — a coin flip, on cv and
holdout alike. This was worth measuring rather than assuming: k=256 was the
largest context ever tested here, the profile was monotonic up to it, and a
band chosen at the edge of its own grid is a band nobody can defend. GPT-2's
1024-*token* window caps usable context at about 730 words (`extract/
context_lm.py` warns and truncates past that), so k=512 is the last point that
fits and the question cannot be pushed further on this backbone.

**The semantic band is `perlayer_gpt2_k256:8`.** Against `gpt2_mean` that is
+0.0551 cv; against the `perlayer_gpt2_k16:8` this file carried until today,
+0.0154 in 9/9. The k=16 store stays on disk as the cross-check that produced
the first joint model.

The recipe that produced this, for the record — extraction is ~3 min per k on
one GPU, not the ~20 min guessed here before, and the sweep is ~15 min per
subject:

    sbatch --array=0-1 --export=ALL,KS="4 256",LAYERS="0-12 0-12",PER_LAYER=1,\
           OUTNAMES="perlayer_gpt2_k4 perlayer_gpt2_k256" \
           scripts/cluster/extract_semantic.sbatch
    sbatch --time=08:00:00 --export=ALL,LAYER_STORE=perlayer_gpt2_k256,\
           TAGSUF=_k256layers scripts/cluster/semantic_sweep.sbatch

### The old recipe, kept because the argument for it still holds

Depth and context can interact, and the k sweep was run at L12, the one layer
least likely to reveal it. Re-check k=4/16/256 at L8 before freezing the band.
This needs new per-layer extractions at those k (only k=16 has one), so it is
~20 min of GPU per k plus a sweep:

    sbatch --array=0-1 --export=ALL,KS="4 256",LAYERS="0-12 0-12",PER_LAYER=1,\
           OUTNAMES="perlayer_gpt2_k4 perlayer_gpt2_k256" \
           scripts/cluster/extract_semantic.sbatch
    sbatch --time=08:00:00 --dependency=afterok:<jobid> \
           --export=ALL,SOURCES="perlayer_gpt2_k4:8 perlayer_gpt2_k16:8 perlayer_gpt2_k256:8",\
           TAGSUF=_kAtL8 scripts/cluster/semantic_sweep.sbatch

`gpt2_mean` is deliberately absent from SOURCES: the sbatch always passes it
as `--baseline-features`, and naming it in both scores the same band twice.

## Llama vs GPT-2: the backbone sweep (2026-10-02)

TRIBE's text backbone is Llama-3.2-3B, so the semantic band was re-swept on the
Llama family. Llama-3.2-1B (16 blocks, 2048-d), -3.2-3B (28, 3072-d), -3.1-8B
(32, 4096-d), k = 256 and 1024 words. Every layer, nine subjects,
`common_stories_all9`, `--min-ev 0.1 --max-repeats 5`, both evals, and
`perlayer_gpt2_k256:8` scored **inside every task** as a paired reference row.
Read with `PYTHONPATH=. python scripts/summarise_model_sweep.py --tags
llama1b_k256 llama3b_k256 llama8b_k256 llama1b_k1024 llama3b_k1024
llama8b_k1024 --profile`. Layer chosen on cv, holdout at that layer:

    model    k      best  depth    cv     vs gpt2  n     holdout  vs gpt2  n
    1B       256    L8    0.50   0.1928  +0.0159  9/9   0.3469  +0.0171  9/9
    3B       256    L13   0.46   0.1938  +0.0169  9/9   0.3483  +0.0185  9/9
    8B       256    L13   0.41   0.1953  +0.0184  9/9   0.3474  +0.0177  8/9
    1B       1024   L8    0.50   0.1932  +0.0163  9/9   0.3489  +0.0191  9/9
    3B       1024   L13   0.46   0.1944  +0.0175  9/9   0.3514  +0.0217  9/9
    8B       1024   L13   0.41   0.1962  +0.0193  9/9   0.3523  +0.0225  8/9

**Switching family is the effect; size within it is small.** Every Llama beats
GPT-2 small in 9/9 subjects on cv, by +0.016 to +0.019 -- about as much as
k=16 -> k=256 bought. 1B -> 8B adds only +0.0030 (k1024, 7/9); 3B -> 8B
+0.0018 (6/9). k=1024 over k=256 is +0.0004 to +0.0009 at matched size: context
is saturated by 256 words here too, despite TRIBE's no-plateau claim.

**The peak sits lower than GPT-2's**: 41-50% of depth against L8/12 = 67%. Same
inverted U, flat-ish around the peak (8B L11-L15 within 0.0025).

Weights: `NousResearch/Llama-3.2-1B` and `NousResearch/Meta-Llama-3.1-8B`
(every file SHA-identical to meta-llama), `unsloth/Llama-3.2-3B` (weights
SHA-identical; tokenizer.json re-serialised, merges equal after format
normalisation). No HF token needed. Models in `~/scratch/hf` on Baobab
(`HF_HOME=$HOME/scratch/hf`), stores in `~/scratch/NL_features` symlinked from
`data/features/`, 25 common stories only (`STORIES=common_stories_all9.json`),
fp32 extraction.

**gpu050 gave wrong numbers.** The three tasks that completed there shifted
every row, the fixed reference rows included, by -0.0007 to -0.0011 cv, while
184 runs elsewhere agree to 1e-8. Re-run with `--exclude=gpu050`; the deviant
outputs are kept as `results/encoding/semantic_sweep/*/_gpu050__*`. The
summariser's reference-row check is what caught it -- keep a shared reference
row in any multi-run comparison.

### DECIDED 2026-10-03: the semantic band is `perlayer_llama8b_k1024:13`

Chosen on cv per the rule. It replaces `perlayer_gpt2_k256:8`, which stays on
disk as the cross-check, and every downstream fit is being redone on it.

**No PCA -- tested, not assumed** (`scripts/cluster/pca_check.sbatch`,
`scripts/summarise_pca_check.py`). Single-band text fits, nine subjects, paired
against the unprojected 4096-d band:

    PCA              variance   cv vs none   PCA ahead   holdout vs none
    1024 comps         0.84     -0.0004        0/9         -0.0005  3/9
    0.90 (1404)        0.90     -0.0003        1/9         -0.0002  ~coin
    256 comps          0.57     -0.0013        0/9         -0.0012  3/9

Worse on cv in essentially every subject *despite* cv flattering PCA (it is
fitted once on all 24 training stories before the outer folds, so it is
transductive there); a coin flip on holdout, where it is clean. Banded ridge
gives each band its own alpha, so 4096 vs 1024 does not need equalising, and
the dual solver's cost is set by TRs, not columns.

**Joint model** (`encoding_joint.sbatch`, TEXT=perlayer_llama8b_k1024:13),
nine subjects, mean over EV>0.1 voxels:

    eval      band     text    audio   joint   joint-text   delta     pref
    cv        llama   0.1965  0.1505  0.2084   +0.0120   +0.0070 9/9  +0.0460 9/9
    cv        gpt2    0.1770  0.1505  0.1882   +0.0113   +0.0059 9/9  +0.0265 9/9
    holdout   llama   0.3535  0.2908  0.3721   +0.0186   +0.0031 7/9  +0.0627 9/9
    holdout   gpt2    0.3297  0.2908  0.3495   +0.0197   +0.0032 6/9  +0.0390 9/9

**What audio adds beyond text did not shrink** (+0.0120 vs +0.0113 cv) against a
text band 0.02 stronger. The prosodic contribution is not something a better LM
already carries -- the cleanest answer yet to "is prosody just lexical
information the text band was too weak to capture". `preference` roughly
doubles; it now has a ~190-configuration selection budget on the text side
against ~96 for audio, so the winner's curse runs toward text on cv. Report
`preference` on holdout.

Permutation cost on this band (20-perm probe): UTS04 2.3 s/perm (was 1.4),
UTS03 10.6 s/perm (was 7.8) -- 1000 x 2 directions ~6 h on the largest mask,
inside the 12 h limit.

### Inference on the Llama band (2026-10-05)

Conditional (13357509) and marginal `both` (13357510), 1000 perms,
`--from-encoding`, nulls saved; group maps in `results/group/
llama8bk1024L13__base_emotionL11__{conditional,marginalfixed}/`, nulls at
`/root/prosody_nulls/llama8b_{conditional,marginal}/`. Marginal refits match
the joint fit to 1e-5 in 9/9. Neither band uses PCA, here or in the GPT-2 runs
compared against.

Within subject (holdout, FDR within EV mask), llama vs gpt2:

    audio|text   +0.0186  20% of voxels   (gpt2 +0.0197  26%)
    text|audio   +0.0813  76%             (gpt2 +0.0587  72%)
    delta        +0.0031   4%, 4/9 subj   (gpt2 +0.0032   9%, 7/9)

Group (61,168 vertices, n>=5):

    statistic      mean     perm FDR   prev FDR   FWE n=9
    audio|text   +0.0196    34,925     23,034     2,076   (gpt2 41,150 / 2,279)
    text|audio   +0.0881    61,133     60,967     2,701
    delta        +0.0123    34,870     22,157     1,451   (gpt2 41,129 / 1,811)
    preference   +0.0674    text ahead 85.8%      FWE 13   (gpt2 +0.0408, 85.2%)

**Why fewer audio voxels at an unchanged effect: the null moved, not the
observed.** Shuffling audio now costs the joint less (null mean -0.007 vs
-0.012 on UTS01/04/09; null SD unchanged at ~0.018) -- banded ridge shrinks a
useless audio band more efficiently beside a stronger text band, so the margin
over the null is ~15% smaller. A calibration effect, not noise.

**Parcels (Desikan) say audio did not lose anything where it lives:**
audio|text *rises* in Heschl (lh 0.086 -> 0.094) and STG (0.049 -> 0.052).
What falls is **text|audio in Heschl (0.024 -> 0.014)** -- Llama predicts
primary auditory cortex *less* than GPT-2 k256 did. Evidence on the open
Heschl question in "integration grows with context": GPT-2's long-context
Heschl signal looks like low-level temporal structure rather than semantic
prediction, since a better LM drops it. A clue, not a test -- the test is the
`wordrate` band in the same fit. Delta falls in Heschl for that reason, and in
MTG/IFG/supramarginal because text|audio jumped (0.060 -> 0.089) and the min is
now audio-bound (the same confound as the k grid: do not write "integration
decreases in IFG"). Banks of STS is where delta grows (rh 0.027 -> 0.030).

### k=1024 survives the onset clock (2026-10-05)

`encoding_onset.sbatch`, `TEXT_FMT=perlayer_llama8b_k{K}:13`,
`TAGPRE=llamaonset`, nine subjects (13377132). Adding the time-since-onset band
barely dents the Llama context gain, unlike GPT-2's (where it halved):

    eval      contrast         text alone      with onset (joint)
    cv        k256 - k16     +0.0222  9/9     +0.0171  9/9
    cv        k1024 - k256   +0.0009  7/9     +0.0011  8/9
    holdout   k256 - k16     +0.0260  9/9     +0.0189  9/9
    holdout   k1024 - k256   +0.0049  8/9     +0.0043  8/9

So k=1024's small lead is not the onset clock, and the band stays
`perlayer_llama8b_k1024:13`. **DECIDED: no onset band in the reported
models** -- it was a check on the k choice, not a nuisance regressor. Downstream submitted 2026-10-05, all
`--exclude=gpu050`: grid joints k4/k16/k256 (13382270/274/309), each with
conditional `_null` and `_marginalfixed` perms chained `aftercorr`;
prediction four-band `TAG=llamapredk1024` (13382312) and interaction
`TAG=llamainterpc20` (13382313), each with a UTS03 20-perm `_llamaprobe`
(13382331/2) before the 1000-perm nulls. The prediction fit first failed in 3 s on every
task: `encoding_prediction.sbatch` requires `K` even when `PRED` and `TAG` are
given -- resubmitted with `K=1024` (13393156, probe 13393157). Interaction
probe: ~21 s/perm on UTS03 (~6 h for 1000); full null `_llamanull` 13393164.

Status 2026-10-06 07:55, from output files: grid conditional `_null` 9/9 at
k4/k16/k256; prediction fit `llamapredk1024` 9/9 both evals, its probe done,
full 1000-perm `_llamanull` submitted (13399681). **Interaction on the Llama
band is still null:** `delta_inter_given_text+audio` has 0 FDR voxels in 9/9
(means -0.0004 to +0.0027, slightly positive in 7/9 where GPT-2's were
negative in 7/9). Note `group_maps.py --contrasts` is a post-step on an
existing group dir: passed with `--perm-dir` it computes only preference and
returns, silently skipping the projection.

### Integration x context on Llama: the GPT-2 grid replicates (2026-10-06)

Group maps `results/group/llama8bk{4,16,256,1024}L13__base_emotionL11__
{conditional,marginalfixed}`, nulls at `/root/prosody_nulls/llama8b_k*_*`.
(Store names differ: k4/k16 are `llama8b_k*` single-layer stores, k256/k1024
`perlayer_llama8b_k*`; same model, layer 13.)

    k       audio|text  text|audio   delta    perm FDR   prev FDR   FWE n=9
    k4       +0.0220     +0.0417    +0.0088    36,693     18,501    1,047
    k16      +0.0218     +0.0599    +0.0097    33,715     18,093      903
    k256     +0.0194     +0.0837    +0.0121    35,849     21,759    1,310
    k1024    +0.0196     +0.0881    +0.0123    34,870     22,157    1,451

Same shape as GPT-2: delta rises with k and saturates past 256; audio|text
erodes as text|audio grows. ROI curve (`roi_context_curve_llama.csv`, lh mean
delta, k4/k16/k256/k1024):

    Heschl       -0.0041 -0.0082  0.0086  0.0136   sig 0% / 1% / 95% / 97%
    STG           0.0092  0.0105  0.0190  0.0214   sig 67% / 66% / 95% / 95%
    precentral    0.0110  0.0164  0.0184  0.0201
    IFG           0.0137  0.0147  0.0117  0.0117   <- falls with the bias
    MTG           0.0112  0.0116  0.0089  0.0093   <- falls with the bias

**The Heschl switch replicates on a different LM family**: text|audio in
Heschl goes from negative at k<=16 to +0.009/+0.014 at k256/k1024, the same
0% -> ~100% transition GPT-2 showed, at about half GPT-2 k256's magnitude
(0.024). So "a better LM drops Heschl" (2026-10-05 parcels) was too strong:
Llama reduces it but the long-context Heschl signal is a property of long
context, not of GPT-2. The `wordrate` test in the prediction fit is still what
decides between temporal structure and semantic prediction. The STG/precentral
growth (interpretable, against the bias) and the IFG/MTG decline
(confounded) are unchanged from GPT-2.

Marginal (`--shuffle-block both`), group means; audio 0.3077 at every k, the
check that the four runs share one design:

    k       text     joint    preference   text ahead   pref FWE n=9
    k4     0.3275   0.3494     +0.0189       70.3%          0
    k16    0.3458   0.3676     +0.0375       79.5%          0
    k256   0.3719   0.3913     +0.0632       85.8%          9
    k1024  (see "Inference on the Llama band": +0.0674, 85.8%, FWE 13)

Interaction group map (`llama8bk1024L13__base_emotionL11__inter_pc20`): mean
+0.0015, perm FDR 0, prevalence 0, FWE 0. Additive, on Llama too.

### Prediction on Llama: the left-STG term is gone (2026-10-06)

`--shuffle-block pred` against the four-band `llamapredk1024` fit (pred =
`surp_pred_llama8b_k1024`, Llama's own surprisal), 1000 perms (13399681),
group map `results/group/llama8bk1024L13__base_emotionL11__pred_given_rest`.
Within subject: 0 FDR voxels in 9/9, means -0.0020 to +0.0007.

**The group counts look larger than GPT-2's and mean less.** Same anatomy
(lh STG 1,371, banks STS 735, rh STG 370, lh MTG/IFG), but split raw vs
null-centred where significant:

                      n      raw Δr    centred   null mean
    llama perm FDR  3,835   -0.0001   +0.0026    -0.0027
    llama prev FDR    662   +0.0000   +0.0034    -0.0034
    llama FWE n=9      69   -0.0021   +0.0009    -0.0030
    gpt2  perm FDR  1,376   +0.0056   +0.0053    +0.0003

On GPT-2 the joint model really scored above `text+audio+wordrate` in left
STG. On Llama the observed Δr is **zero**. Vertices clear the test only
because the shuffled pred band *lowers* the joint (null mean -0.003), so pred
beats its own permuted copy but adds nothing over the model without it. The
Draper-Stoneman test is valid for its H0 (pred exchangeable). That H0 is just
not "pred improves prediction". **Write it as: with a Llama text band, surprisal
and entropy add nothing measurable. The GPT-2 left-STG term was prediction
information GPT-2 small's hidden state lacked.** Do not quote the 3,835.

General lesson: whenever a conditional-null map has a mean near zero, report
raw Δr next to `mean_centred`. A significant count with raw Δr ≤ 0 means the
null was shifted down, not that the band adds anything.

### Llama downstream: clusters, k contrasts, partition (2026-10-08)

Run locally from the existing nulls (`/root/prosody_nulls/run_llama_downstream.sh`).
Outputs: `llama8bk{4,16,256}L13__..._conditional/clusters/` (k1024 was done
2026-10-07), `context_contrast_llama/`, `integration_grid_llama/`,
`context_clusters_report_llama.html`, `llama8bk1024L13__..._partition/`.

Group-null TFCE FWE vertices (family 61,168), and z>3.09 extent for delta:

    k      a|t     t|a     delta   | z3.09 delta  clusters
    4    39,162  52,793  35,974    |   16,325        24
    16   37,941  53,081  34,784    |   15,708        28
    256  37,681  56,641  37,612    |   19,306        28
    1024 35,682  56,627  35,622    |   19,879        27
    (gpt2 k256: 38,982 / 56,246 / 38,920 | 27,784, 30)

Paired sign-flip k contrasts (TFCE FWE, `up` = larger at the larger k):

    pair       delta              audio|text          text|audio
    4:16     -0.0007   0 / 0    -0.0085   0 / 4,003  +0.0203  8,552 / 0
    4:256    +0.0019 3,468 / 0  -0.0208   0 / 12,097 +0.0613 13,763 / 0
    16:256   +0.0026 3,120 / 0  -0.0123   0 / 9,903  +0.0410 12,766 / 0
    256:1024 +0.0006 1,515 / 0  +0.0009 1,816 / 0    +0.0013      0 / 0
    (gpt2: 4:256 +0.0064 4,355 up; 16:256 +0.0073 4,015 up; 256:512 3,376 DOWN)

Same structure as GPT-2: integration grows from short to long context while
audio|text erodes and text|audio grows, so only regions where delta rises
against that bias are interpretable. The growth is about a third of GPT-2's
in mean. 256 -> 1024 does not reverse the way GPT-2's 256 -> 512 did; it adds
a small rise in delta (1,515 TFCE vertices) carried by audio|text, not text.

Partition (k1024): unique text 61,133, unique audio 34,925, both 34,895,
audio-only 30 vertices. Wherever audio adds, text adds too.

### Word rate does not explain the text band (2026-10-07, read 2026-10-08)

`encoding_wordrate.sbatch` (13417892, `bands_text-audio-wordrate__llamawordrate`)
and `--shuffle-block text` null keeping `audio+wordrate` (13417894,
`..._shuffle_text__llamawrnull`), 9/9 both evals, 1000 perms. Mean over EV mask:

    eval      t|a      t|(a+wr)  ratio   wr|(t+a)
    cv       +0.0582  +0.0589   1.012   +0.0011  9/9 > 0
    holdout  +0.0817  +0.0829   1.014   +0.0016  9/9 > 0

Text given audio+wordrate is FDR-significant in 65-85% of each subject's mask.
Mask-wide, word rate absorbs none of the text contribution.

**Heschl, read 2026-10-09: word rate does not explain it either.** Group map
`results/group/llama8bk1024L13__base_emotionL11__text_given_audio_wordrate`
(nulls `/root/prosody_nulls/llama8b_wordrate/`): mean +0.0902, perm FDR 61,134,
FWE 2,722. Desikan, t|(a+wr) against t|a from the k1024 conditional run:

    parcel                  t|a      t|(a+wr)   sig (wr)
    lh transversetemporal  0.0136    0.0129      98.1%
    rh transversetemporal  0.0166    0.0147     100.0%
    lh superiortemporal    0.0428    0.0455     100.0%
    lh bankssts            0.0629    0.0658     100.0%

Heschl loses 5% (lh) / 12% (rh) and stays ~100% significant; everywhere else
t|(a+wr) is at or slightly above t|a (a different three-band fit, so
fourth-decimal moves are fit noise). So the long-context Heschl signal is not
word-arrival timing. Low-level temporal structure beyond word rate (syllable
rate, phrase boundaries) is not excluded by this test.

### openSMILE as the audio band, and the audio bands against each other (2026-10-12)

Jobs 13451040-42 (Llama k1024 L13 x `opensmile`: fit, conditional, marginal)
and 13452773-76 (`scripts/cluster/encoding_bands.sbatch`, two-band fits
`emotion=perlayer_base_emotion:11 opensmile=opensmile` and
`opensmile=opensmile avd=emotion_avd`, each with `--shuffle-block each`),
1000 perms, nulls saved, all 9/9. Group maps (n>=5 family 61,168):

    run / statistic              mean    perm FDR  prev FDR  FWE n=9
    osm marginal  audio        +0.1191   46,563    41,013    2,812
    osm cond.     audio|text   +0.0108   18,677    15,042      543
                  text|audio   +0.2679   61,166    61,165    2,974
                  delta        +0.0095   18,672    15,015      224
    emo vs osm    emo|osm      +0.1869   61,161    61,125    3,287
                  osm|emo      -0.0017    1,857     1,585       10
    osm vs avd    osm|avd      +0.0460   42,023    30,767    2,367
                  avd|osm      +0.0178   29,417    20,749      447

- **Integration with openSMILE is Heschl-centred**: audio|text and delta peak
  in transversetemporal (0.061 / 0.035, 100% sig), then STG. Within subject
  the conjunction is significant in 8/9 subjects (6.3% of voxels) against 4/9
  (4%) with the emotion band, despite audio|text being half as large.
- **openSMILE adds nothing beyond the emotion band.** Within subject 0 FDR
  voxels in 9/9; at group level the 1,857 vertices have raw Δr +0.0051 (centred
  +0.0354), almost all right Heschl. The emotion band contains what openSMILE
  explains.
- **Affect adds beyond acoustics**: AVD|openSMILE positive in 9/9 subjects,
  group 29,417 vertices, strongest in lh pars orbitalis/triangularis, banks of
  STS and superior frontal -- the valence anatomy, not auditory cortex.
  openSMILE|AVD is the reverse: Heschl, STG, sensorimotor.

Clusters for every paper map: `/root/prosody_nulls/run_clusters_paper.sh`,
using `group_clusters.py --stats` (added 2026-10-12: any statistic, no
conjunction) for everything except the two text/audio conditional runs.
All seven done 2026-10-12 (AVD marginal re-run by hand after the session
running it was killed). `PYTHONPATH=. python3 scripts/paper_table.py` collects
every paper map -- mean, perm FDR, TFCE FWE, z>3.09 extent clusters and the
largest ones by peak Glasser area -- into `results/group/paper_table/
paper_table.{csv,md}`, in the paper's order. Nothing is re-tested there.
openSMILE|emotion keeps 5 z>3.09 clusters (A1/belt, 5,944 TFCE vertices) at a
raw gain of +0.005: not an effect, see above.

## Inference: what the irony project did, and what carries over (2026-09-08)

`../Clean_Irony/permutation_test.py` and `diagnostic_permutations.py` are the
methods reference for this project's statistics. Read them before writing
anything new here. Verified against the source, not remembered:

**Draper–Stoneman conditional nulls.** `--shuffle_block {both,text,audio}`:
`both` shuffles every feature block and gives the null for "r > 0"; `audio`
shuffles the audio columns while text stays aligned and gives the null for
`Δr_audio|text`; `text` mirrors it. Block modes force
`include_mod=['text_audio']` — only the *combined* model is refit, because the
whole point is that the joint model keeps its full dimensionality and can
still overfit with the shuffled block's columns. That is what makes the null
absorb the nesting bias.

**The statistic is a conditional contribution, not a max.**

    observed  Δr_audio|text = r_joint − r_text
    null_i    Δr_audio|text = r_joint(audio shuffled)_i − r_text

`r_text` is the same observed constant on both sides, so the p-value reduces
to `P(r_joint_shuffled ≥ r_joint)`; subtracting it only keeps the effect size
on an interpretable scale. **Integration is then the *conjunction* of the two
conditional contributions** — audio adds beyond text AND text adds beyond
audio — never a test of `r_joint − max(...)` against zero.

**Alphas are fixed, not re-searched.** `optimize_alpha=False` with
`valphas=_subset_valphas(...)` loads the per-voxel alphas from the observed
fit, so a permutation refits weights only. Without this the alpha search runs
inside every permutation and the whole thing is unaffordable.

**Derived columns are rebuilt after shuffling.** `_rebuild_interactions`
recomputes `semantic_* x prosody_*` as the product of its *shuffled* parents,
so the interaction follows the permuted design instead of silently keeping
the aligned one.

**Null calibration was checked, not assumed.** `diagnostic_permutations.py`
produces, per test: p-value histogram (calibration), observed-vs-null
separation, QQ plot of −log10 p, example per-voxel nulls, effect sizes split
by significance, and null-std vs observed r (homogeneity). FDR is
`statsmodels.fdrcorrection` at α=0.05. Reproduce this figure here; it is the
supplementary that answers the reviewer question before it is asked.

### What must change when porting it

- **Shuffle blocks of TIME, not rows.** Irony shuffles trials within
  participant, which is exchangeable in an event-related design. This is
  continuous naturalistic listening with an HRF and heavy temporal
  autocorrelation: a free row shuffle destroys that autocorrelation, narrows
  the null and makes the test **anticonservative**. Use the existing
  `block_permutation_index` with blocklen ≥ HRF width (10 TRs = 20 s).
- **Shuffle the raw features, then rebuild the FIR delays** — not the delayed
  design matrix, or alignment leaks across block boundaries through the delay
  copies.
- **Two levels here, one there.** Irony pools every participant into one ridge
  (`participant_ids`, shuffle within participant), so the permutation *is* the
  inference. This project fits one model per subject, so within-subject
  permutation and the group claim are separate questions.

### Why not a group test in the Caucheteux & King style

They had hundreds of subjects; a group-level nonparametric test there has both
resolution and random-effects validity. Here n=9, and that closes the route:

    exact sign-flip / signed-rank / sign test: 2^9 = 512 arrangements
      -> smallest attainable one-sided p = 1/512 = 0.00195

BH at that floor needs ≥69 voxels pegged at the minimum p before it rejects
anything inside the EV>0.1 mask, and **≥3,169 whole-brain** — more than the
~1,776 voxels that carry explainable signal at all. Whole-brain group FDR at
n=9 is arithmetically impossible, not merely expensive. Consequences:

- Primary inference is **within subject**, where there are 8,683 training TRs.
- For a group map, use **max-statistic sign-flip FWE** (PALM/randomise style),
  which needs no per-voxel p below 1/512 because the threshold is on the null
  of the max: the 95th percentile sits at rank 486/512, 26 values in the tail.
  Coarse but valid.
- Otherwise state the group claim as consistency — "significant in k of 9" —
  which is what the sweeps already support (9/9 for L8, 9/9 for context; a
  sign test on 9/9 is exactly p = 0.00195).

### Delta is NOT retired — the conjunction is how it gets tested

`delta = r_joint − max(r_text, r_audio)` stays the quantity of interest. What
cannot be done is permuting *every* block and recomputing the same formula:
that null is "no encoding anywhere", under which all three correlations go to
~0, while the observed delta is a difference of two *large* correlations
carrying the nesting bias. Comparing them tests "is there any signal", so a
purely unimodal voxel passes. The H0 that is wanted — "the joint model
explains no more than the best single modality" — is not the H0 that shuffling
everything produces.

The Draper–Stoneman conditional nulls are how to model the right H0, and an
exact identity makes the connection rigorous rather than approximate:

    r_joint − max(r_text, r_audio) ≡ min(r_joint − r_text, r_joint − r_audio)
    delta                          ≡ min(Δr_audio|text, Δr_text|audio)

(verified numerically, exact to 0.0). So `delta > 0` **iff both conditional
contributions are > 0**, and the conjunction of the two one-sided DS tests is
an *intersection–union test* of `H0: delta ≤ 0` — valid at level α with **no
multiplicity correction between the two tests** (Berger 1982). Conservative,
which is the right direction here.

Practically: report `delta` as the effect-size map, and take its per-voxel
p-value as `max(p_audio|text, p_text|audio)`, then FDR across voxels on that.
Each component p comes from its own DS null, each of which models the correct
H0 by construction. Nothing about the header's definition changes.

## The final joint model: which bands

Each band is chosen on **cv** over its own sweep. As of 2026-09-09:

    prosody    perlayer_base_emotion 11     +0.064 over openSMILE, 9/9
                                      (decided 2026-09-10; see above)
    semantic   perlayer_gpt2_k256 layer 8   +0.0551 over gpt2_mean, 9/9
                                      (decided 2026-09-15; k at L8, see above)

> **SUPERSEDED 2026-10-03: semantic is now `perlayer_llama8b_k1024:13`**
> (+0.0193 cv over `perlayer_gpt2_k256:8`, 9/9). Every GPT-2 number below is
> the previous band; see "Llama vs GPT-2" and "DECIDED 2026-10-03".

**Both bands are now on one set of masks, so they can finally be put side by
side** — the thing this file forbade until 2026-09-11. Same nine subjects, same
24 common stories, same `--min-ev 0.1 --max-repeats 5` EV masks, one runner
each:

    band                         cv      holdout
    text   perlayer_gpt2_k256:8      0.1769    0.3298
    audio  perlayer_base_emotion:11  0.1505    0.2908
    preference = r_text - r_audio   +0.0264   +0.0390
    text ahead in                      9/9       9/9

**Text is ahead in every subject, on both evaluations.** A sign test on 9/9 is
p = 0.002, the 1/512 floor, which is as strong as any group claim in this file
gets. The margin is +0.0264 cv, about 18% of the audio band's own score.

This is the second version of this table and the first one was materially
different: at `perlayer_gpt2_k16:8` it read +0.0111 cv with text ahead in 8/9,
UTS04 preferring audio on both evals and UTS05 on holdout. Fixing the semantic
band did not just raise the text number, it removed every subject-level
exception. Worth remembering when reading any "prosody wins here" claim from
before 2026-09-15: the text band it was measured against was 0.0154 short.

Note the two selection budgets push the *other* way (see below): audio was
chosen over ~96 configurations against text's ~19 (now ~45 with the k x layer
grid), so the cross-validated audio number is still the more optimistic of the
pair.

These are marginal scores from single-band fits. They are not `preference` as
the paper will report it, which comes per voxel out of the joint model — but
they set the scale.

This supersedes the L10-vs-L11 mismatch that used to be recorded here. The
corrected sweep puts the peak at 9-11 (emotion) and L11 (robust), and the 45 old
holdout runs that used `base_emotion_L11` / `base_robust_L18` were on the
misaligned grid and are void regardless of which layer they named.

**DONE (`2ba552b`).** `run_encoding` takes the `store:layer` source syntax
`run_semantic_sweep` already has (`perlayer_base_emotion:11`,
`perlayer_gpt2_k16:8`) instead of a flat store per layer. One mechanism, no
duplicated stores, and it removes the class of error that produced that
mismatch. The store's own `layers` attribute is read rather than assumed,
because the two writers disagree about what index *i* means.

**CLOSED 2026-09-11: the semantic numbers are on the current EV masks.** Every
semantic sweep used to predate `--max-repeats 5` (2026-09-09), so UTS01-03 were
scored over masks built from 10 repeats while every prosody number came from 5 —
1,776 voxels against 6,555 for UTS01 — and this section carried a standing ban
on comparing a text number with an audio number. The depth sweep at k=16 has
been re-run on the current masks (9/9, both evals) and the ban is lifted; the
old results stay at `results/encoding/semantic_sweep_oldmask_20260909/`.

**Selection budgets are unequal, and that biases `preference` on cv.** The audio
band was chosen over ~96 configurations (4 stores x ~24 layers), the text band
over ~19. A max over more candidates carries more winner's curse, so the
cross-validated audio score is the more optimistic of the two. The held-out story
is unaffected, since neither selection touched it — one more reason to report
`preference` on holdout only, and to say in the methods how many configurations
each band was selected over.

## The joint model, first run (2026-09-15) — and what delta does on holdout

`scripts/cluster/encoding_joint.sbatch` fits the two chosen bands and the joint model
through one runner, nine subjects, both evals, ~40 min per subject on one GPU
with no `eigh` fallbacks. The first run used `perlayer_gpt2_k16:8` before the k
question was settled; it is kept as a cross-check and the numbers below are from
it. The `perlayer_gpt2_k256:8` re-run has since completed (9/9, both evals,
36-46 min per subject) and is the fit the permutation section below tests, so
treat the table here as the superseded cross-check rather than the reported
model. What did not move between the two is the structure, which is why it is
recorded at all.

    eval       text    audio    joint     delta    pref   delta>0 voxels
    cv       0.1616   0.1505   0.1768   +0.0047  +0.0111      60-78%
    holdout  0.3101   0.2908   0.3353   -0.0011  +0.0194      37-64%

    delta > 0 in 9/9 subjects on cv, 4/9 on holdout

**The joint model beats both unimodal bands in every subject on both evals** —
0.1768 against 0.1616 and 0.1505 on cv. The nesting works.

**But mean per-voxel `delta` is ~0 on holdout, and that is not a contradiction.**
`delta = r_joint - max(r_text, r_audio)` takes the max *per voxel*, and the max
of two noisy estimates is biased upward by roughly the noise scale. On 291 TRs
the per-voxel standard errors are ~4x the cross-validated ones, so the max
absorbs the joint model's advantage and delta collapses. On cv, where the same
voxel is scored over 8,683 TRs, delta is positive in 9/9.

So this file's standing line — "under banded ridge `delta >= 0` almost by
construction" — is a statement about the *fit* nesting, not about a *noisy
estimate* of r on one story. It holds on cv and fails on holdout. Do not read
the holdout delta map as evidence against integration; read it as the reason the
permutation test exists, because the Draper-Stoneman null carries exactly the
same upward bias in its own max and therefore cancels it. Measured on the probe:
UTS04's `delta_observed` mean is 0.00097 through `run_permutation` against
0.0010 through `run_encoding` — the same quantity, two code paths.

**The conditional contributions are large where delta is small.** UTS04, holdout:

    delta_audio_given_text   mean +0.0410   (audio beyond text)
    delta_text_given_audio   mean +0.0210   (text beyond audio)
    delta (their per-voxel min)     +0.0010

Both bands add substantially to the other, but rarely in the *same* voxel — which
is what a small min against two large components means, and it is a result about
the anatomy of the effect rather than a weakness of it.

### What a permutation costs here, measured

The `--n-perms 20` probe (`scripts/cluster/permutation.sbatch`, TAGSUF=_probe) exists
because the cost was never measured and is mask-dependent by an order of
magnitude. It is:

    UTS04   1,706 voxels   1.4 s/perm    1000 x 2 directions ~ 46 min
    UTS03  16,395 voxels   7.8 s/perm    1000 x 2 directions ~ 4.3 h

So `--n-perms 1000` fits inside the 12 h `shared-gpu` limit on every subject with
room to spare, and there is no need to trade p-value resolution for wall clock.
Run the probe first anyway when the design changes: a timed-out task writes
nothing at all, and the probe costs two minutes.

A 20-permutation probe reports `n_significant: 0` everywhere. That is correct,
not a failure — the smallest attainable p is 1/21 = 0.048 and nothing survives
FDR across thousands of voxels. It is only a plumbing check.

## The inference: Draper-Stoneman conditional nulls, 1000 permutations (2026-09-15)

Run on the final bands (`perlayer_gpt2_k256:8` against `perlayer_base_emotion:11`),
nine subjects, `--shuffle-block conditional --n-perms 1000 --blocklen 10`,
alphas and band weights reused from the joint fit via `--from-encoding` so the
observed statistic is the published one rather than a second search that agrees
with it. Wall clock 51 min to 5 h 01 per subject, scaling with mask size, all
nine COMPLETED. FDR q < 0.05 within each subject's EV mask.

    subject  voxels    audio|text          text|audio          delta (conjunction)
    UTS01      6555   +0.0214  2020  31%  +0.0688   5374  82%  +0.0079   933  14%
    UTS02     11254   +0.0061  1029   9%  +0.0687   8672  77%  -0.0026     0   0%
    UTS03     16395   +0.0068  1118   7%  +0.0796  12280  75%  -0.0012   333   2%
    UTS04      1706   +0.0285   666  39%  +0.0535   1221  72%  +0.0122   307  18%
    UTS05      1736   +0.0370  1002  58%  +0.0387   1119  64%  +0.0054   363  21%
    UTS06      3192   +0.0318  1018  32%  +0.0663   2452  77%  +0.0073   144   5%
    UTS07      2489   +0.0194   699  28%  +0.0453   1669  67%  +0.0014   240  10%
    UTS08      2608   +0.0043   119   5%  +0.0514   1455  56%  -0.0081     0   0%
    UTS09      4092   +0.0222  1103  27%  +0.0558   3069  75%  +0.0068   348   9%
    MEAN      50027   +0.0197        18%  +0.0587         75%  +0.0032         5%

**Both bands contribute conditionally in 9 of 9 subjects.** Audio adds beyond
text somewhere in every subject, and text adds beyond audio somewhere in every
subject. Neither modality is redundant given the other, which is the first thing
this project needed to establish and the thing a single-band comparison cannot
show.

**Text's conditional contribution is about three times audio's**, +0.0587 against
+0.0197 on average, and larger in every subject individually. That is the same
ordering as `preference` and it is now the conditional version of it: text does
not merely predict better on its own, it adds more once the other band is already
in the model.

**Integration is significant in 7 of 9 subjects and is spatially restricted.**
The conjunction clears FDR in about 5% of explainable-signal voxels on average,
ranging from 0% (UTS02, UTS08) to 21% (UTS05). So integration is real and
localised rather than general: most voxels that carry signal are explained well
by one band with the other adding nothing detectable on top.

**What limits the conjunction is the audio term, not the text term.** The
intersection-union test needs both components significant in the same voxel, and
audio clears FDR in 18% of voxels against text's 75%. The delta map is therefore
bounded above by the audio map almost everywhere, and the three subjects with the
weakest audio contribution (UTS08 5%, UTS03 7%, UTS02 9%) are exactly the three
with zero or near-zero integration voxels. Do not read a low delta count as weak
integration without reporting the audio-conditional map beside it; the constraint
is visible in the data rather than inferred.

**The negative delta means are not a contradiction.** UTS02, UTS03 and UTS08 have
mean delta below zero while still showing significant conditional contributions,
because `delta` takes the per-voxel max of two noisy estimates and that max is
biased upward on a 291-TR story. The permutation is what handles it: the null
carries the identical bias, which is why voxels still clear FDR in a map whose
mean is negative. This is the measured version of the argument in the joint-model
section above.

Read the maps back from:

    results/encoding/perlayer_gpt2_k256L8__perlayer_base_emotionL11__shuffle_conditional/permutation/<SUBJ>/

`delta_significant.npy` is the integration map, `delta_audio_given_text_*` and
`delta_text_given_audio_*` the two components. Note the standing caveat that
`run_permutation` FDR-corrects within the EV ROI and then pads to full brain with
p = 1, so anything downstream must not correct a second time.


### The null is calibrated, measured rather than assumed (2026-09-23)

`stats/diagnose_permutation.py`, on the 1000-draw conditional run, nine
subjects. Every saved draw is by hypothesis a sample from H0, so draw *i* is
treated as if it were the observed statistic and given a p-value against the
other 999 with the estimator the real test uses:

    statistic                 FPR    nominal   KS dev
    delta_audio_given_text   0.0469    0.05     0.029
    delta_text_given_audio   0.0497    0.05     0.005
    delta (the IU rule)      0.0068    0.05     0.215

Per subject, `audio|text` spans 0.0438–0.0491 and `text|audio` 0.0494–0.0499 —
no subject deviates, including the three whose masks differ by an order of
magnitude. And `delta` comes out at 0.0068 against 0.05 nominal, which is the
intersection–union test's conservativeness *measured*: the max of two p-values
is stochastically larger than uniform, so the rule must under-reject, and it
does, by a factor of 7.

**Be honest about what this panel can and cannot see.** Null draws are
exchangeable with each other by construction whatever the block length, so this
validates the `(b+1)/(m+1)` estimator, the tie handling and the IU rule — not
the assumption that the block shuffle preserves the data's autocorrelation.
That assumption is what the next section tests, which is why both exist.

Figure and numbers: `results/encoding/k256L8__base_emotionL11__conditional_diagnostics/`.

### `--blocklen 10` sits on the plateau, and blocklen 1 would not have (2026-09-23)

Five block lengths on three subjects spanning mask size (UTS03 16,395 voxels,
UTS06 3,192, UTS04 1,706), 200 permutations each. **Read the null's width, not
the significant-voxel counts** — at 200 draws the smallest attainable p is
1/201 and FDR resolution dominates the counts, which is why UTS03 reports zero
significant voxels at every blocklen ≥ 5 while the published 1000-draw run
reports 1,118. Mean SD of the null:

    statistic    subject     bl1     bl5    bl10    bl20    bl40   observed
    audio|text   UTS03     0.0112  0.0142  0.0149  0.0151  0.0147  +0.0068
                 UTS04     0.0173  0.0219  0.0232  0.0235  0.0227  +0.0285
                 UTS06     0.0174  0.0215  0.0221  0.0219  0.0212  +0.0318
    text|audio   UTS03     0.0614  0.0750  0.0770  0.0778  0.0778  +0.0796
                 UTS04     0.0408  0.0522  0.0552  0.0552  0.0553  +0.0535
                 UTS06     0.0590  0.0702  0.0717  0.0726  0.0705  +0.0663

**blocklen 1 narrows the null by 20–25%**, which is the free row shuffle
destroying the autocorrelation exactly as this file predicted when it ported
the irony permutations — and it is visible in the counts too, where bl1 hands
UTS03 2,133 significant `audio|text` voxels against 0 at bl5 on identical
resolution. bl5 is still 4–5% narrow. **bl10, bl20 and bl40 agree to within
1–3% and not monotonically, i.e. noise**, so 10 TRs is on the plateau and the
plateau starts just below it. The published inference stands, and the HRF-width
argument for the floor is now a measurement.

## CLOSED 2026-09-23: integration grows with context — in temporal cortex only

The k × integration grid (k = 4, 16, 256, 512 against `perlayer_base_emotion:11`,
nine subjects, conditional and marginal, 1000 permutations, group maps under
`results/group/k*L8__base_emotionL11__*`). At group level:

    k       delta mean   perm FDR          FWE (n>=9)
    k4       +0.0105    39,553  64.7%     1,237/3,337
    k16      +0.0108    36,177  59.1%     1,046/3,337
    k256     +0.0137    41,129  67.2%     1,811/3,337   <- max
    k512     +0.0130    40,615  66.4%     1,762/3,337

Integration rises with context and **saturates between 256 and 512, the same
place the encoding score saturates** — one more reason `perlayer_gpt2_k256:8`
is a ceiling rather than an edge.

**The group mean averages two opposite behaviours, and only one of them is
interpretable.** `scripts/roi_context_curve.py` (Desikan, written to
`results/group/roi_context_curve.csv`), mean `delta`, left hemisphere:

    ROI                  k4      k16     k256    k512
    HeschlTransverse  -0.0004  -0.0018  0.0241  0.0234   <- 0% -> 100% sig
    STG                0.0097   0.0128  0.0268  0.0257
    precentral         0.0107   0.0139  0.0185  0.0185
    supramarginal      0.0115   0.0132  0.0172  0.0164
    IFG                0.0169   0.0157  0.0145  0.0139   <- falls
    MTG                0.0140   0.0129  0.0092  0.0086   <- falls
    inferiorparietal   0.0059   0.0050  0.0032  0.0030

Meanwhile `delta_audio_given_text` **falls with k in every single ROI** (lh STG
0.0713 -> 0.0464, lh IFG 0.0225 -> 0.0141) while `text|audio` rises. That is
precisely the confound the script's docstring names in advance: "context
absorbs prosody" and "a stronger text band absorbs prosody" predict the same
thing. So:

- **STG, Heschl, precentral, supramarginal grow against the bias** and the
  growth is real. There the min is bound by the *text* term, and more context
  releases it.
- **IFG, MTG and inferior parietal fall with the bias** and cannot be read.
  There the min is bound by the *audio* term, which erodes mechanically as the
  text band strengthens. Do not write "context reduces integration in IFG".
- **Heschl is the striking cell**: 0% of covered vertices significant at k4 and
  k16, **100% at k256**, because `text|audio` there goes from 0.3% to 100%.
  With 256 words of context the semantic band starts predicting primary
  auditory cortex. Either top-down prediction or long-context embeddings
  carrying low-level temporal structure — settle it before writing a sentence
  about it.

Coverage caveat, and it is not small: IFG is covered at 42% (lh) and 33% (rh)
with ~5.9 subjects reaching it on average, and precentral at 7–11%. Those rows
describe five subjects' periphery. `n_covered` and `coverage` are in every row
of the CSV for exactly this reason.

## The prediction band: surprisal, entropy, and the word-rate trap (2026-09-23)

`extract/surprisal.py` writes three TR-aligned quantities off GPT-2's LM head —
surprisal, entropy and word rate — plus `pe_gpt2_k<K>`, the 768-d
prediction-error vector `E[emb | w_<t] - emb(w_t)` that is the naturalistic port
of `../Clean_Irony`'s context–statement incongruence. Extracted 2026-09-22
(job 13114585, five tasks, k = 0, 4, 16, 64, 256): **84 stories in all ten
stores**, files counted rather than SLURM states trusted.

Grid verified against the semantic stores before anything was fit:
`surprisal_gpt2_k256`, `pe_gpt2_k256` and `perlayer_gpt2_k256` have identical
`n_TRs` on every story checked, no NaNs, and the five all-zero rows sit at the
head where `[TR_PAD + trim : -trim]` removes them — the post-realignment
signature.

### The stored band is mostly word rate, and at k=0 it is word rate exactly

Both prediction columns are word-level signals resampled onto TR times with
`lanczosinterp2D`, so their amplitude tracks how many words arrived. On the
design slice, 84 stories, 28,088 TRs:

    r with word_rate     k0      k4      k16     k64     k256
    surprisal          +0.976  +0.841  +0.766  +0.742  +0.740
    entropy            +1.000  +0.958  +0.935  +0.930  +0.928

**`r(entropy, word_rate) = 1.000` at k=0 is not a rounding artefact**: with no
context the per-word entropy is a constant, and a constant resampled by word
arrival *is* the word-rate signal. The k=0 store is rank-deficient by
construction.

This also corrects the claim in `extract_surprisal.sbatch`'s own header that
k=0 correlates with k=256 "at only r = 0.59, so contextual surprisal is
demonstrably not word frequency". Per TR the raw figure is **0.805**; 0.59 is
what you get *after* partialling out word rate (measured: 0.569). The
conclusion survives, the number as written does not — and the gap between them
is the whole problem.

### So word rate gets its own band

Banded ridge assigns one alpha per *band*, not per column, so a three-column
store fit as one band shrinks the nuisance and the signal together and no
contrast can separate them. `scripts/split_surprisal_bands.py` writes

    wordrate           1 column, once, shared by every k (verified bit-identical
                       across k on all 84 stories — the script raises otherwise)
    surp_pred_k<K>     2 columns, [surprisal, entropy] at that context

and then `r_joint − r_wordrate` is what prediction adds beyond word rate: the
same conditional-contribution logic as the text/audio permutations, one level
down, and the only form in which a surprisal claim survives the reviewer who
says "that is just word rate".

**The split is needed for the k comparison, not only for the interpretation.**
word_rate is identical across k, so fit as one band every k would be handed the
same strong nuisance and k=0 would be handed the most of it. The argmax would
land on the least contextual model for the least contextual reason. Split,
k=0 becomes what it was meant to be: the no-context floor.

Surprisal and entropy stay together — they are the same distribution's first
and second summaries and their partial correlation is 0.555 at k256 with word
rate removed. Split them further only if something asks which of the two
carries the effect.

### What is running, and what is deliberately not

`scripts/cluster/encoding_surprisal.sbatch` ran 2026-09-23, one task per subject, the whole k
grid inside each task, two bands (`wordrate`, `pred`) and three models
(`wordrate`, `pred`, `joint`), cv and holdout, `--min-ev 0.1 --max-repeats 5`.
Choose k on cv by `r_joint − r_wordrate`.

**This is the selection sweep, not the reported model.** The reported claim is
what prediction adds beyond the bands already carried
(`perlayer_gpt2_k256:8` and `perlayer_base_emotion:11`) — a separate and far
more expensive fit at the chosen k, and a permutation after it.

The `pe_gpt2_k<K>` vectors are held back on purpose. They are a band of a
different size and a different argument, and mixing them into the scalar grid
would confound "which context length" with "scalar or vector".

Note the sbatch lives at `scripts/cluster/` here and at `scripts/` on Baobab —
the three-drawer reorganisation is local and unpushed, so the cluster still has
the flat layout. Anything submitted there takes the flat path.

### DONE 2026-09-24: prediction adds beyond word rate in 9/9, at every k

Nine subjects x five context lengths x both evals, `common_stories_all9`,
`--min-ev 0.1 --max-repeats 5`. Read with
`PYTHONPATH=. python3 scripts/summarise_surprisal_sweep.py`. Mean r over each
subject's EV mask, averaged over the nine:

    k      wordrate     pred     joint   joint-wr    n      holdout(joint-wr)
    0        0.0640    0.0740    0.0739   +0.0099   9/9        +0.0077  8/9
    4        0.0640    0.0716    0.0798   +0.0158   9/9        +0.0094  7/9
    16       0.0640    0.0698    0.0750   +0.0110   9/9        -0.0002  6/9
    64       0.0640    0.0724    0.0773   +0.0133   9/9        +0.0041  6/9
    256      0.0640    0.0754    0.0798   +0.0158   9/9        +0.0020  5/9

**Surprisal and entropy add to word rate in every subject at every context
length.** That is the first thing this had to establish and it is not a given:
the band is two columns against a nuisance they correlate with at +0.74 to
+0.98.

**The split paid for itself twice over.** `wordrate` reads 0.0640 on every row,
which is the check that these rows are one comparison — and `pred` *alone*
ranks k=0 second (0.0740, above k4, k16 and k64), exactly the artefact the
split was built to prevent. Marginal selection would have promoted the store
whose entropy column is word rate with r = 1.000.

**cv is an exact tie between k=4 and k=256** — 0.01581 both, to five decimals,
with the per-subject argmax at k256 in five subjects and k4 in four, and
`k256 - k4` a coin flip (+0.0001, 5/9). Holdout prefers k4 (+0.0094 against
+0.0020) but holdout does not select here, and this file has the receipt: over
36 prosody sweeps the cv and holdout argmaxes agreed in 1, and selecting on
holdout would have inflated the reported score by +0.0083.

**DECIDED: k=256, and the tie is broken on interpretability, not on score.**
The semantic band is `perlayer_gpt2_k256:8`. Matching its context makes
`r_joint - r_(text+audio+wordrate)` a contrast between the *head* and the
*hidden state* of one model reading one context — the AVD precedent exactly,
where the affective band was chosen because it is the head of the checkpoint
the encoding already uses. At k=4 a positive result would confound
"prediction adds something" with "short context is a different signal from
long context", which is not the question. It is also the harder test: at
matched context the scalar is maximally redundant with the 768 dimensions it
was computed from, so a positive result there is worth more and a null is
clean.

**The dip at k=16 is reproducible and unexplained.** Every one of the nine
subjects scores k16 below both k4 and k64 on cv — 9/9, on a shared design and
one seed, so it is a property of the `surp_pred_k16` features and not sampling
noise. Do not smooth it over as a wobble; it is 0.005 deep, a third of the
effect. Nothing in the extraction obviously singles k16 out (k0, k4 and k16
share one forward-pass regime, k64 and k256 another), so this is recorded as
an open oddity rather than a finding.

### Prediction adds nothing beyond text + audio + word rate (2026-09-24)

Four-band fit at k=256 (`bands_text-audio-pred-wordrate__predk256`, 9/9 both
evals) and `--shuffle-block pred`, 1000 perms: `delta_pred_given_text+audio+
wordrate` has **0 FDR-significant voxels in 9/9 subjects**, means -0.0007 to
+0.0017. Surprisal/entropy beat word rate alone but not the carried bands.

**At group level it is not nothing** (`results/group/k256L8__base_emotionL11__
pred_given_rest/`, 61,168 covered vertices): perm FDR 1,376 (2.2%),
prevalence FDR 525, FWE 22 of the n>=9 family. Small (+0.0056 mean where
significant) and **left-lateralised in superior temporal cortex**: of the
perm-significant vertices 642 lh STG, 329 lh banks STS, 105 lh MTG, only 66
rh STG, plus ~80 in lh pars opercularis/triangularis. Prevalence is 99% lh
STG + banks STS; all three tests agree there. The classic surprisal network,
visible only by pooling. Write it as "a small, localised unique contribution
in left STG/STS", not as a null.

### Surprisal is already in the audio band (2026-09-28)

Stimulus-side check before any `audio x surprisal` interaction band
(`scripts/surprisal_prominence.py --k 256`, 84 stories, 28,088 TRs on the
design slice, out-of-sample r, folds by story, alpha on an inner split):

    target            opensmile 88d    base_emotion:11 1024d
    word_rate            0.759              0.930
    surprisal            0.682              0.820
    surprisal | wr       0.189              0.581   (R2 0.34)
    entropy | wr         0.092              0.463   (R2 0.21)

**Named prosodic prominence barely carries surprisal** once word rate is
out: partial r median 0.038, max 0.20, and the top features are formant
amplitudes and voicing rates, not F0 or loudness. Aylett & Turk-style
accenting is not visible at TR resolution.

**The learned audio band does carry it**: a third of the rate-independent
surprisal variance is linearly decodable from `emotion:11`. Most likely
lexical identity leaking through a speech model's mid-stack (function words
are short and reduced), not prosody. Two consequences:

- `audio x surprisal` would be partly `audio x g(audio)` -- a nonlinearity of
  the audio band. If the interaction is built, modulate by the part of
  surprisal the audio *cannot* see: `s_perp = s - s_hat(audio)`, cross-fitted
  by story. That is also the theoretically right modulator (the top-down
  component). Linear orthogonalisation only; nonlinear leakage remains.
- Part of why `pred` adds nothing may be the audio band, not only the text
  band. The four-band null does not say which absorbed it.

Word rate at r = 0.93 from the audio band is worth remembering when reading
any audio-vs-text contrast.

## CLOSED 2026-10-02: text x audio interaction adds nothing — combination is additive

Band `inter` = pairwise products of the top 20 PCs of `perlayer_gpt2_k256:8`
and `perlayer_base_emotion:11` (400 columns, PCA fit on the 24 training
stories only; `scripts/make_interaction_band.py`). Three-band fit
(`bands_text-audio-inter__interpc20`, `scripts/cluster/encoding_interaction.sbatch`,
2-3 h per subject), then `--shuffle-block inter`, 1000 perms, kept model
`text+audio`, alphas from `--from-encoding`. pc50 (2,500 columns) was probed
on UTS04 and was worse than pc20 on both evals, so it was not run further.

    subject  voxels   mean    FDR   p<.05 unc   cv joint - text+audio
    UTS01     6555  -0.0015    0      3.6%          -0.0007
    UTS02    11254  -0.0005    0      6.7%          -0.0007
    UTS03    16395  -0.0008    0      3.0%          -0.0006
    UTS04     1706  -0.0018    0      1.2%          -0.0006
    UTS05     1736  -0.0026    0      1.3%          -0.0009
    UTS06     3192  +0.0003    0      2.9%          -0.0005
    UTS07     2489  -0.0020    0      2.5%          -0.0005
    UTS08     2608  +0.0000    0      3.7%          -0.0006
    UTS09     4092  -0.0028    0      1.1%          -0.0006

**0 FDR voxels in 9/9, cv below `text+audio` in 9/9**, and uncorrected
p<.05 rates at or under chance. Group (`results/group/k256L8__base_emotionL11__
inter_pc20`, 61,168 vertices): mean -0.0011, perm FDR 0, prevalence FDR 0,
4 of 3,337 FWE vertices — isolated single vertices with |mean| <= 0.0034, one
of them negative, i.e. the variance floor at work, not a cluster. Unlike `pred`,
pooling does not reveal anything. The alpha grid was not binding: the
inter band's alpha has a median of ~1e10.4, so ridge switches the band off,
and the ~5% of voxels where its penalty is comparable to text/audio did not
survive either.

So within what a low-rank multiplicative term can express, **text and audio
combine additively** — the integration measured by `delta` is two bands
adding in the same voxels, not a gain-modulation of one by the other.
Caveat for the paper: a null for 20x20 PC products is not a null for every
non-linearity.

Permutation output and nulls: `/root/prosody_nulls/bands_text-audio-inter__shuffle_inter__null/`.

## AVD: the per-dimension question, answered with p-values (2026-09-17)

The affective sections above could only report **counts** — "9/9 subjects" —
because `stats/run_permutation.py` was text/audio by construction. It no longer
is, and the three dimensions now have per-voxel nulls. 1000 permutations, nine
subjects, `--from-encoding results/encoding/bands_arousal-dominance-valence`, so
the observed statistic is the published fit rather than a second search.

### `run_permutation` takes N named bands

`--band NAME=STORE` and `--models`, delegating to `run_encoding`'s own
`resolve_band_stores` / `resolve_model_bands` so a permutation and the fit it
tests cannot disagree about what a band is. New `--shuffle-block each`: one
Draper-Stoneman null per band, that band shuffled with every other held aligned.
The kept model is the `+`-joined subset named exactly as run_encoding names it
(`arousal` -> `dominance+valence`), so `r_kept` is read from the fit.

**The two-band path is byte-identical** — `resolve_design_bands` without
`--band` returns `MODEL_BANDS` itself. Three things were two-band by assumption
rather than by design and all three would have failed silently:

- `observed["delta"] = joint - max(text, audio)` was unguarded, and is what the
  first AVD probe died on. `delta` is now computed only when
  `{text, audio, joint}` are all present: with three affective dimensions there
  is no "the other modality" to max against, and synthesising one would redefine
  this project's headline statistic under its own name.
- The `--from-encoding` consistency check compared only the `text`/`audio`
  stores against the fit's metadata, so an N-band run could have been aimed at
  the wrong fit with every guard passing. It now checks every declared band.
- `--shuffle-block both` **used to be** the expensive mode here (it refit all
  seven models with fresh alpha searches: 22 min to 3 h 22 per subject) while
  `each` reuses band weights (26 min to 5 h 25). It no longer searches — see
  "Every permutation reuses the published alphas" below — and now costs under a
  minute of compute per subject.

That first probe reported `COMPLETED` in 1:32 having written **nothing**, and
its log had been overwritten because two probes shared `logs/perm_%a_out`. Both
rules in this file earned their keep: probe when the design changes, and check
the output files rather than the SLURM exit states.

### Every permutation reuses the published alphas (2026-09-21)

**The rule, now without exception: search once, on the real data, per model —
then freeze for inference.** Each model gets its own hyperparameters when its r
is measured, and no model gains or loses capacity when it is tested.

`--shuffle-block both` was the one mode that broke it, and the reason recorded
here was wrong. This file said it "cannot reuse the published fit" because its
null shuffles test-set predictions that `run_encoding` does not save. The first
half is true and the second does not follow: needing a refit is not the same as
needing a second alpha search. **Every model's band weights were already on
disk** — `text_deltas.npy`, `audio_deltas.npy`, `joint_deltas.npy` and all seven
of the AVD run's — and `load_observed_fit` read only the joint's while
`fit_banded_fixed` computed test predictions at `banded.py:350` and threw them
away.

So: `fit_banded_fixed` gained `return_predictions`, `load_observed_fit` reads
every model's weights, and `both` refits each model with **its own** published
weights held fixed to recover its predictions. Measured on UTS04 against the
k256 joint fit:

    model    refit mean r    published    max |diff|
    audio       0.3160        0.3160       2.2e-06
    joint       0.3695        0.3695       8.8e-06
    wall clock  0.8 min/subject  (was 22 min to 3 h 22)

**The joint agrees to 1e-06, which was not the expectation.** `WeightedKernelRidge`
with conjugate gradient converges to the same solution as `MultipleKernelRidgeCV`
at the same deltas, so the estimator gap that forced the Draper-Stoneman path to
use `ds["observed"]` is numerically negligible here. The observed r is still taken
from the refits rather than from `*_corrs.npy`, because a permutation test may not
straddle two estimators even when they agree — and the run logs both, so the gap
is visible on every job instead of assumed.

Why fixing is not merely cheaper but more correct: re-searching on each shuffle
lets the model discover that the shuffled band is now useless and shrink it away,
so the null describes a *well-tuned model on noise* while the observed statistic
came from a model tuned to exploit a band that helps. The two sides stop sharing
a capacity to overfit, and that capacity is exactly what has to cancel for the
test to measure the band's contribution rather than its column count.

The counter-argument deserves stating in the methods rather than being ignored:
the textbook recipe for permuting a *pipeline* is to redo everything the pipeline
does, hyperparameter selection included. That is right when H0 concerns the
procedure's generalisation. It is wrong here because the statistic is a difference
between **nested** models, which is only interpretable at constant capacity.

Cost of the mistake, for the record: nothing. `conditional` and `each` always
reused, so no published inference used a searched null. Only the two `both` runs
did — `perlayer_gpt2_k256L8__..._marginal` and
`bands_arousal-dominance-valence__marginal` — and both are being re-run under
`__marginalfixed` rather than in place, so the old and new can be compared before
anything is replaced.

### What the second alpha search actually cost, measured (2026-09-21)

Keeping the old runs instead of overwriting them is what made this measurable.
UTS04, AVD, old `__marginal` against the new `__marginalfixed` and against the
encoding fit both claim to test:

    model                 old      new      fit    old - fit
    arousal            0.1320   0.1320   0.1320     +0.0000
    dominance          0.1275   0.1275   0.1275     -0.0000
    valence            0.0061   0.0061   0.0061     +0.0000
    joint              0.1227   0.1191   0.1191     +0.0036   <-
    arousal+dominance  0.1134   0.1132   0.1132     +0.0002
    arousal+valence    0.1355   0.1357   0.1357     -0.0002
    dominance+valence  0.1250   0.1256   0.1256     -0.0006

**The disagreement is zero for single-band models and grows with the number of
bands.** With one band the weight simplex is a point, so two independent alpha
searches land in the same place to four decimals. With a simplex to sample,
`random_search` draws different points on different runs: 0.0002-0.0006 at two
bands, **+0.0036 at three**.

So the old AVD marginal was testing a joint model *better than the published
one*, while the conditional analysis beside it used the published fit through
`--from-encoding`. The two AVD tables in this file came from two different joint
models. The new run reproduces the fit for all seven.

No conclusion moves — the joint falls further below arousal alone (0.1320), which
strengthens "arousal and dominance predict better alone and add nothing unique".
What changes is that this file's own explanation for a nested joint scoring below
its own subset, "a randomised band-weight search", stops being an assertion:
**two draws on identical data differ by 0.004 at three bands.**

The published conditional inference is untouched, having never made a second draw.

### Only valence contributes uniquely, and the marginal run says why

`delta_<dim>_given_<rest>`, holdout, mean over the EV mask and FDR-significant
voxel count:

    subject   arousal|D+V       dominance|A+V      valence|A+D
    UTS01      -0.0006   918      -0.0065   679     +0.0122   923
    UTS02      -0.0121   914      -0.0152   662     +0.0203  1478
    UTS03      -0.0038  2195      -0.0074  1949     +0.0294  3600
    UTS04      -0.0065   307      -0.0166   220     +0.0059   168
    UTS05      -0.0200   386      -0.0208   301     +0.0562   749
    UTS06      +0.0004   352      -0.0034   296     +0.0104   512
    UTS07      -0.0216   144      -0.0127   167     +0.0165   438
    UTS08      -0.0083   224      +0.0088   388     +0.0201     0
    UTS09      -0.0266   196      -0.0240   170     +0.0291   279

**Valence positive in 9/9, arousal and dominance negative in 8/9** (sign test
p = 0.002, the 1/512 floor). This resolves the cv-vs-holdout sign disagreement
recorded above — in the holdout direction, per voxel, against a real null
instead of a count.

The marginal run inverts the picture and explains it:

    model      UTS01     UTS04     UTS09     valence FDR-significant voxels
    arousal   +0.0948   +0.1320   +0.0626    0 in 7 of 9 subjects
    dominance +0.0857   +0.1275   +0.0683    (only UTS05, 347)
    valence   +0.0342   +0.0061   +0.0109
    joint     +0.0960   +0.1227   +0.0735

**Arousal and dominance predict two to three times better on their own and add
nothing unique; valence predicts worst alone and is the only one that adds.**
That is exactly what effective rank 2 looks like at arousal x dominance
r = 0.963: the collinear pair is mutually redundant, valence is the orthogonal
direction, and only the conditional test can see it. The earlier instruction to
"report valence" is now measured rather than inferred.

The one-sided test means arousal and dominance still have FDR-significant voxels
against a negative mean — those are voxels where they genuinely do add.

### At group level, on fsaverage (`results/group/avd__{conditional,marginal}/`)

61,168 vertices covered by >=5 subjects, 3,337 by all nine.

    conditional   mean r      perm FDR      prevalence FDR    FWE (n=9)
    arousal      -0.0080    8,038  13.1%    9,210  15.1%          0
    dominance    -0.0096    7,013  11.5%    8,234  13.5%          0
    valence      +0.0215   29,091  47.6%   21,364  34.9%        186

    marginal      mean r      perm FDR      prevalence FDR    FWE (n=9)
    arousal      +0.0757   25,838           21,322            1,760
    dominance    +0.0730   24,897           19,930            2,253
    valence      +0.0362        0            5,352               13
    joint        +0.0894   36,693           31,862            2,476

**The inversion is complete and it is visible in one pair of rows.** Valence
marginally clears *zero* vertices under the group permutation and is the weakest
band on its own; conditionally it covers four times the cortex of either
collinear dimension and is the only one with a positive mean or any FWE-
significant vertices at all. Arousal and dominance are the mirror image.

Valence's marginal 0-against-5,352 split between the permutation and prevalence
is not a bug: the prediction-shuffle null for a *marginal* r on a 291-TR story
has spread ~1/sqrt(291) = 0.059, wider than valence's observed 0.0362, while
prevalence pools weak per-subject evidence across nine people. Two valid tests
of two different hypotheses; report both.

The valence map is bilateral superior temporal and temporo-parietal with strong
prefrontal clusters. 32 figures across the four group directories.

## The group maps: fsaverage, averaged over subjects, with p-values (2026-09-16)

`scripts/group_maps.py`. Projects each subject's observed map **and its 1000-draw
null** to fsaverage, then produces three group tests per statistic. Six
statistics: `text`, `audio`, `joint` from a `--shuffle-block both` run, and
`delta_audio_given_text`, `delta_text_given_audio`, `delta` from the conditional
one. Both runs are on the final bands, 1000 perms, `--min-ev 0.1 --max-repeats 5`.

The September permutation discarded its nulls (`--save-null` was never passed),
so both were re-run. The conditional re-run reproduces the published per-subject
table **exactly** — same seed, same numbers to four decimals and the same voxel
counts. The `both` run was, at the time, an independent check: it could not
use `--from-encoding` and refit with its own alpha search, and it landed on
`r_text = 0.3297` / `r_audio = 0.2908` against the joint fit's
`0.3298 / 0.2908`. That independence is gone by design as of 2026-09-21 — the
mode now reuses the published weights and agrees to 1e-06 rather than to four
decimals — so the numbers in the table below come from the last run that still
searched. They are being re-run; expect movement in the fourth decimal only.

    statistic              mean r      perm FDR      prevalence FDR    FWE (n=9)
    text                  +0.3488   61,168 100.0%   61,163 100.0%    3,337 100.0%
    audio                 +0.3077   61,151 100.0%   61,138 100.0%    3,337 100.0%
    joint                 +0.3683   61,168 100.0%   61,163 100.0%    3,337 100.0%
    delta_audio_given_text +0.0196  41,150  67.3%   32,854  53.7%    2,279  68.3%
    delta_text_given_audio +0.0607  61,156 100.0%   61,080  99.9%    2,829  84.8%
    delta (conjunction)    +0.0137  41,129  67.2%   32,693  53.4%    1,811  54.3%

    preference = r_text - r_audio   +0.0411, text ahead at 84.7% of vertices

**The group picture is far stronger than the within-subject one.** Integration
cleared FDR in ~5% of voxels within subject and in only 7 of 9 subjects; at group
level the conjunction is significant in 67% of covered vertices. That is what
pooling buys, and it is why per-subject significance must **not** be intersected
across subjects before averaging — a 9-way conjunction tests "an effect in every
subject", a different and much stronger hypothesis than the one wanted, and the
weakest subject alone would zero the map.

### The three tests, and what each licenses

- **`p_perm`** — group permutation. The nine subjects share `--seed 42` and have
  the same 8,683 training TRs (verified in every `meta.json`), so null draw *i*
  is the identical block shuffle for all nine and the per-draw group mean is a
  valid null. Resolution 1/1001. **Fixed effects**: about these nine people and
  this stimulus, not a population.
- **`p_prevalence`** — binomial count (Allefeld, Görgen & Haynes 2016). Under H0
  a permutation p is uniform, so the count of subjects significant at uncorrected
  alpha is exactly Binomial(n_v, alpha); 3 of 9 already gives p = 0.008, 4 gives
  6e-04. The per-subject p is recomputed *on the surface* from the projected
  null, never by projecting the volume p-map — interpolating p-values is
  smoothing and the result stops being uniform.
- **`p_signflip_fwe`** — max-statistic over the 2^9 = 512 sign vectors. The only
  **random-effects** test here, and the weakest. Each subject is centred on its
  own null mean first; a raw `Delta r` is not symmetric about zero under H0,
  which is the whole reason the nulls had to be kept.

`delta` is the intersection-union of its two components, so its p is the
per-vertex max of the two, computed within each family, no correction between
them.

### Calibration, checked rather than assumed

`text_given_audio` is significant at 100.0% of covered vertices, which is the
sort of number that is usually a broken null. It is not. Binned by how many
subjects were individually significant at a vertex:

    k of 9 subjects sig    vertices    group mean r    group significant
    0                        11,701       -0.0006            9.1%
    1                        12,031       +0.0045           49.9%
    2                         8,876       +0.0105           85.8%

(audio|text; the group test tracks the per-subject evidence monotonically and
does not reject where there is none.) For text|audio only **5** vertices have
zero subjects significant and the group test rejects at **0.0%** of them — the
effect really is positive at 99.8% of covered vertices.

### Two things that bite

**Thin coverage is biased DOWNWARD, not upward — the open question in
`project_to_fsaverage`'s docstring, now measured.** That docstring warns that the
subjects reaching a vertex are the ones whose EV mask included it, so thin
vertices are "*selected* for high r" and the group mean is "biased upward exactly
where it is least supported". It is the reverse:

    n_subj   vertices      text     audio   delta_audio|text
       5       24,491    +0.3235   +0.2697      +0.0118
       7       10,605    +0.3690   +0.3381      +0.0249
       9        3,337    +0.4140   +0.4034      +0.0405

    Spearman rho(mean r, n_subjects): text +0.461, audio +0.526, delta +0.391

Vertices all nine reach are core auditory/language cortex where everyone has
signal; vertices five reach are periphery, where r is *lower*. So `--min-subjects`
is conservative in the right direction, and the variance argument for a floor was
the only one that ever needed to hold.

**Consequence for the FWE numbers:** the n>=9 family is not a random subsample,
it is the strongest-signal ~1% of cortex (mean r 0.414 against 0.324 at n=5).
"68% of the n>=9 family" describes the core and must **not** be read as 68% of
cortex. Report perm/prevalence for extent, FWE for the core.

**Coverage is the binding constraint, and it is not statistical.** Vertices
reached by *k* subjects: 58,792 by one, 24,491 by five, and only **3,337 of
327,684 by all nine**. Nine different `--min-ev 0.1` masks barely overlap once
projected. 61,168 vertices clear the `--min-subjects 5` floor. A fuller group map
needs a lower `--min-ev` **at fit time**; nothing downstream can recover it.

**A max-statistic FWE is destroyed by thinly-covered vertices, and this cost a
run to find.** Two compounding failures, both measured:

1. *Plain max-t is set by the smallest dispersion, not the largest effect.* Nine
   subjects who happen to agree closely give t -> infinity as sd -> 0. On the
   first run the smallest attainable FWE p was **0.047** and nothing survived
   anywhere. A simulation with known ground truth: t = 80.4 at no-effect
   vertices whose variance had collapsed, against t = 14.5 where the effect was.
   Fixed with a variance floor at the 25th percentile of observed sd — one
   constant taken from the unflipped data and applied identically to every flip,
   so the test stays exact. Detection went 0.7% -> 31.3% with zero false
   positives at every setting.
2. *The family matters more than the statistic.* Same data, same fixed test,
   only the coverage floor changed:

        family          audio|text        text|audio      delta
        n>=5  61,168 vtx     19  (0.03%)      4  (0.01%)      0
        n>=7  20,672 vtx    229  (1.11%)    674  (3.26%)      0
        n>=8  10,067 vtx  2,980 (29.60%)  6,161 (61.20%)  1,561
        n>=9   3,337 vtx  2,279 (68.29%)  2,829 (84.78%)  1,811

   The p-values are valid at every floor; what changes is whether the test can
   see anything. `--fwe-min-subjects` defaults to **9** — the only family with
   homogeneous degrees of freedom — while `--min-subjects` (5) still governs the
   mean, permutation and prevalence maps.

### `preference` needs its own test, and the sign test cannot deliver it

`preference = r_text - r_audio` has **no permutation of its own**, and cannot
borrow one: the prediction-shuffle null behind `text` and `audio` is H0 "this
model does not predict at all", under which *both* correlations go to ~0 and so
does their difference. It tests for signal, not for one band beating the other.

What is available without a new null is a **paired** test across subjects, which
is the right shape anyway — the same subject contributes both numbers, so the
difference removes that subject's data quality. `--contrasts` computes two:

    mean preference   +0.0408      text ahead at 85.2% of covered vertices
    sign-flip FWE     27 vertices of the 3,337-vertex n>=9 family
    exact sign test   0 vertices survive BH

**The sign test rejecting nothing is arithmetic, not absence of effect** — and it
is this file's own n=9 argument, instantiated. Subjects favouring text, per
vertex: 798 have 0 of 9, 18,473 have 5, and only **320 have 9 of 9**. The exact
sign test floors at 1/512 = 0.00195, and BH over 61,168 vertices needs

    0.00195 <= 0.05 * k / 61,168   ->   k >= 2,389 vertices pegged at the floor

against the 320 that exist. Uncorrected, p < 0.05 at 19,315 vertices (31.6%).
So report `preference` as an **effect-size map** with the FWE overlay, never as a
sign-test map, and say why.

### Figures

`scripts/visualisation/plot_group_maps.py` renders every statistic in a group directory to an
fsaverage flatmap, thresholded and unthresholded, over curvature.

    python scripts/visualisation/plot_group_maps.py --root results/group/<run>
    python scripts/visualisation/plot_group_maps.py --root results/group/<run> --test none

Colour follows the job the map does. Magnitude maps (`r_*`, every `delta`) are
**one hue, light to dark** — they are non-negative by construction, so a
diverging scale would spend half its range on impossible values and imply a
midpoint that does not exist. `preference` is the only signed map and gets **two
hues with a neutral grey midpoint**, forced symmetric about zero, so that
"neither band ahead" reads as absence of colour and neither modality is made to
look stronger by an asymmetric scale. No rainbows, no hue at the midpoint.

Two things learned rendering them:

- **Both sequential ramps are truncated to start at a tinted step.** The pale end
  of a full `Reds`/`Blues` ramp is the same lightness as the curvature
  background, so the weakest suprathreshold vertices read as "no data" rather
  than "small effect".
- **ROI outlines need Inkscape**, which pycortex shells out to and which is not
  installed here; the script warns and drops the outlines rather than raising
  from inside `quickflat`. `apt-get install inkscape` restores them.

What the figures show, and it is the expected anatomy: integration is bilateral
and centred on superior temporal cortex with prefrontal and parietal clusters;
`preference` is red (text ahead) almost everywhere with a blue (audio ahead)
band running through superior temporal cortex — early auditory cortex being the
one place the prosodic band wins.

### The same maps elsewhere: inflated views, parcels, CONN (2026-09-21)

A flatmap shows every vertex at once, which is what extent needs, and no
recognisable anatomy, which is what a talk needs. Three scripts read a
`results/group/<run>` directory and write beside it, so one run directory
holds every rendering of the same numbers:

    results/group/<run>/
      *.npy               the maps, p-maps, coverage, null stacks
      figures/            pycortex flatmaps        plot_group_maps.py
      figures_inflated/   fsaverage lat/med views  plot_group_inflated.py
      tables/             per-parcel CSVs          plot_group_inflated.py
      export/conn/        CONN .surf.nii           export_group_conn.py
      export/mni/         MNI152 2 mm .nii         export_group_mni.py

Nothing is re-projected or re-tested anywhere below. The maps are already on
fsaverage and already carry p-values; these only draw or re-container them.

**`plot_group_inflated.py`** reimports the colormaps, labels and default tests
from `plot_group_maps.py`, so the two cannot drift. It also writes a parcel
table (Desikan by default, `--atlas a2009s` for Destrieux, or any
`<hemi>.<name>.annot` under fsaverage/label): mean effect, % of covered
vertices significant, and `pct_covered` — which answers "what about region X"
with "not covered" rather than "no effect" when that is the truth.

*Not* the pycortex ROI outlines, and it is worth recording why so nobody
installs Inkscape expecting them to help: the fsaverage `overlays.svg` in this
database is the Huth-lab retinotopy set (V1-V4, hMT, FFA, PPA, IPS0-5, VO1/2)
plus a few sulci. For an auditory/language result it names almost nothing.

**The parcel tables found something the flatmaps hid: the two conditional
contributions are not in the same place.** Desikan means over covered
vertices, `delta_*__parcels_aparc.csv`:

    audio beyond text                  text beyond audio
    lh transversetemporal  +0.086      lh parahippocampal    +0.097
    rh transversetemporal  +0.080      rh superiorparietal   +0.092
    rh superiortemporal    +0.052      lh precuneus          +0.086
    lh superiortemporal    +0.049      rh isthmuscingulate   +0.083
    rh precentral          +0.044      rh superiorfrontal    +0.078

Heschl -> STG -> sensorimotor against precuneus, parahippocampal, superior
parietal, superior frontal. That is the mechanism behind the conjunction being
restricted to ~5% of voxels within subject: the min is small because the two
components overlap little, not because either is weak. The conjunction sits
where they do overlap — STG bilaterally, banks of the STS, Heschl, 97-100% of
those parcels' covered vertices significant.

The AVD inversion is anatomical too. Arousal and dominance marginally peak in
*primary auditory* cortex (transversetemporal +0.318/+0.294) — acoustics
wearing an affective label — while valence's unique contribution is left
inferior frontal and lateral orbitofrontal (parsorbitalis +0.0525, 90%
significant; parsopercularis, parstriangularis, lateralorbitofrontal).

**`export_group_conn.py` is the right way into CONN, not a volume.** CONN's
surface space *is* fsaverage, so the vertices transfer with no resampling:
`42 x 83 x 47 = 163,842` exactly, which is what identifies those dimensions
(`conn_surf_dims(8)`), both hemispheres stacked lh-then-rh along the third
axis as `[42 83 94]`, MATLAB column-major reshape. **Not** `[42 83 47 2]`,
which the first version wrote (fixed 2026-09-23, checked against CONN's own
`conn_surf_write` / `conn_surf_dimscheck`): `dimscheck` reads only the first
three dimensions, so that file is loaded as a volume, not a surface. Vertex counts in the written files reproduce the published table
exactly, which is a free check that nothing was scrambled.

Two conventions cannot be tested from here — hemisphere order and reshape
order — and getting either wrong writes a file of exactly the right size full
of noise. So each output folder also gets `_check_curvature.surf.nii`,
fsaverage's own curvature in the same layout: open it first, and if it renders
as a folded brain with light gyri and dark sulci the layout is right and every
other file in the folder is right with it.

**`export_group_mni.py` is the fallback, and it is a forward conversion, not a
recovery.** These maps were never volumetric: each subject is fitted in their
own functional space and the group average exists only on fsaverage, because
the nine functional volumes are in nine different spaces — the LeBel pipeline
is subject-native plus pycortex and never normalises to MNI. The script goes
`mri_surf2vol` through fsaverage's ribbon, applies the constant MNI305->MNI152
affine, and resamples onto the 79x95x79 2 mm SPM/CONN grid (`--grid` to match
another). Consequences: nothing exists outside the cortical ribbon, and **peak
coordinates off these volumes are not trustworthy** the way a volumetric
analysis's are — two resamplings and one affine. Render with them; quote
numbers from the surface or the parcel tables.

### Practical

    sbatch --export=ALL,TEXT=perlayer_gpt2_k256:8,N_PERMS=1000,\
           SHUFFLE=conditional,SAVE_NULL=1,TAGSUF=_null scripts/cluster/permutation.sbatch
    sbatch --export=ALL,TEXT=perlayer_gpt2_k256:8,N_PERMS=1000,\
           SHUFFLE=both,SAVE_NULL=1,TAGSUF=_marginal scripts/cluster/permutation.sbatch

    python scripts/group_maps.py --perm-dir <run>/permutation --out results/group/<name>

    # then, on the same directory, any of:
    python scripts/visualisation/plot_group_maps.py     --root results/group/<name>   # flatmaps
    python scripts/visualisation/plot_group_inflated.py --root results/group/<name>   # + parcels
    python scripts/visualisation/export_group_conn.py   --root results/group/<name>   # CONN
    python scripts/visualisation/export_group_mni.py    --root results/group/<name>   # volumes
    python scripts/visualisation/conn_figures.py       --root results/group/<name>   # CONN mosaic8
                                                        # -> publication SVG/PNG/PDF

`conn_figures.py` runs after `export_group_conn.py`: it launches MATLAB itself
(switching the Windows default printer away from the Horizon "(vdi)" driver
that crashes `matlab -batch`, and restoring it), prints each map with CONN's
own mosaic8 printer via `print_group_conn.m`, and lays the result out with
the neural_integration `figtools` into `<run>/figures_conn/`. Never launch
MATLAB with `-WindowStyle Hidden` for this: the camera views come out wrong.

`scripts/cluster/permutation.sbatch` now takes `SAVE_NULL` and passes
`--from-encoding` in **every** mode, `both` included. Nulls are ~1.3 GB for both runs
and are kept off OneDrive. Projection needs FreeSurfer on `PATH`
(`/usr/local/freesurfer/7.4.1/bin`) and runs **locally** — the 2.4 GB pycortex
database is at `data/ds003020/derivative/pycortex-db` and is not on the cluster.
Cost: ~20 min per run, 122 ms per projected map, 9,000 maps per statistic pair.

Each statistic saves `__mean`, `__p_perm(_fdr)`, `__p_prevalence(_fdr)`,
`__p_signflip_fwe`, `__n_subjects`, `__k_significant`, `__fwe_family` and
`__centred_stack` (9 x 327,684, ~12 MB). The stack exists so
`--recompute-fwe` can retune the family or the variance floor in seconds instead
of re-projecting for 20 minutes.

## Code review, 2026-09-09 — what was wrong, and what was done

A multi-lens audit of the pipeline. Everything below was verified against the
code and, where a number is quoted, measured on the real data. **All of it is
fixed in `e659e3a`** unless marked otherwise; the diagnosis is kept because the
numbers say how much each mattered, and because two of them changed results
that are already in this file.

### Touched published numbers — FIXED

**`preprocess.py:147` z-scores the held-out design with the TEST story's own
statistics.** There is a `fitted_pca` escape hatch, no `fitted_scalers` one, so
ridge weights learned in pooled-train units are applied to per-story-
standardised columns and column *j* is silently re-weighted by
`sd_train_j / sd_test_j`. Measured over the 24 common stories vs
`wheretheressmoke`:

    band                        median   p95      max     >2x
    opensmile                     1.28   9.54   1058.7   15.9%
    perlayer_base_emotion L10     1.05   1.19      1.6    0.0%
    perlayer_gpt2_k16 L8          1.04   1.13      1.3    0.0%
    gpt2_mean                     1.03   1.10      1.2    0.0%

So the bands the final joint model uses are barely touched, and **`--eval cv`
is unaffected entirely** (one scaler over the whole training design) — every
layer and context selection stands. The damage is confined to **openSMILE on
holdout**, whose score is degraded by an artifact the neural bands do not feel.
Consequence: the holdout Δ-vs-openSMILE columns overstate the neural bands, and
a future `preference` on holdout with an openSMILE audio band would be biased
toward text. Note this is the opposite of the rule CLAUDE.md already states for
the fine-tuning stack ("validation scalers come from training").

**Fixed:** `build_design` takes `fitted_scalers` mirroring `fitted_pca`, and
all four callers pass it. The training path is byte-identical — `_standardise`
reproduces `npp.zscore` to `max|diff| = 0.0`, constant columns included — so
nothing selected on cv moves. **The openSMILE holdout numbers already in this
file were computed before the fix** and are the ones to re-run if the paper
quotes a holdout Δ-vs-openSMILE.

### Would have broken the next thing that runs — FIXED

**`scripts/project_to_fsaverage.py:260` averages masked-out voxels as real
zeros.** `run_encoding.py:355` scatters unfitted voxels back as `np.zeros`, and
`coverage()` is `project(ones) > 0` — purely *anatomical*, with no knowledge of
the EV mask (`voxel_mask.npy` is in `SKIP_SUFFIXES` and never projected). So
`np.nanmean` averages 79,350 structural zeros in. Measured on UTS01: mean r over
fitted voxels **+0.0335**, over the full array **+0.0007** — a **45.7x
dilution**, uneven across the map because subjects disagree about which voxels
pass EV. The module docstring (lines 17-21) promises exactly the opposite. Only
3 fsaverage files exist so far, so nothing reported depends on it yet.
Compounding it: the two fill conventions in the repo disagree — `run_encoding`
fills `0.0`, both sweeps fill `np.nan` — and `project()` converts NaN to 0.0
anyway, so honest missing-data marking is destroyed on the way to the surface.

**Fixed:** `project_masked` normalises by the mask's own interpolation weight,
recovering the weighted mean over fitted voxels and returning NaN where none
reach. Whether a map is mask-scattered is decided by *looking at its values*
(everything outside the mask is 0 or NaN), not by its name — which is what makes
it robust to the two writers padding differently. `ev.npy` is defined
everywhere and so still projects plainly. The three existing `*_fsaverage.npy`
files predate this and should be regenerated with `--overwrite`.

**`stats/analysis.py` double-dips on `preference`.** `compute_contrasts:99`
thresholds on `max(r_text, r_audio, r_joint) > min_r` and then reports
`preference = r_text - r_audio` over exactly those voxels. Conditioning on the
max inflates |preference| within the selected set and tilts the counts toward
whichever band has the larger sampling variance. `min_r=0.05` is below the
per-voxel SE on a 291-TR story, so the gate is mostly noise. An independent
mask is already on disk and unused: `voxel_mask`, which uses no model output.
Also `winner_map:138` overwrites the semantic/prosodic labels with
"integrative" wherever `delta > 0` unthresholded — the exact call the
permutation machinery exists to replace — while `delta_significant` sits unread.
And `group_summary:212` stacks subject maps of different voxel counts (81,126 to
109,469), so it either raises or silently writes a one-subject "group" mean.

**Fixed:** selection defaults to the EV mask (`--selection min_r` keeps the old
behaviour and warns); `winner_map` is preference-only and integration moved to
`integration_map`, which takes `delta_significant` via `--permutation-dir` and
warns loudly when it has to fall back to an unthresholded delta; and
`group_summary` refuses a native-space group mean across different voxel spaces,
pointing at `project_to_fsaverage` instead, while NaN-ing each subject outside
its own selection.

### The EV mask: what it does and does not break

**It does NOT invalidate the permutation.** The permutation is conditional on Y
and EV is a function of Y alone — it never sees the features or the model — so
conditioning on it leaves exchangeability intact and the test exactly valid.
Selecting on measurement reliability buys power; it does not move the null.

**It is unstable at 5 repeats**, which is a different problem from the
"estimated twice as precisely" note elsewhere in this file. UTS01 as its own
control, same subject, same story, only the repeat count changed:

    all 10 repeats   1,776 voxels   mean ev 0.158
    repeats 1-5      6,555 (3.7x)   mean ev 0.156
    repeats 6-10     1,024 (0.58x)  mean ev 0.169

Mean EV is unchanged, so the bias correction works; the *variance* swings mask
size by 6x between two halves of the same data. UTS04-09's masks are therefore
unreliable rather than uniformly bigger or smaller, and "mean r over EV>0.1
voxels" is not the same quantity across subjects.

**`noise_ceiling()` returns the wrong ceiling.** `sqrt(EV)` is the *single-
repeat* ceiling, but r is scored against the *mean* of repeats. At the observed
mean rho=0.158 the attainable ceiling is **0.696 at n=5 and 0.808 at n=10**,
while `cv.py:107` returns **0.398 for both** — about half the true value, and
identical across subjects, so it corrects none of the imbalance
`stats/analysis.py:24` advertises it for. Live at `analysis.py:91-93`.

**Fixed:** `noise_ceiling(ev, n_repeats)` applies Spearman-Brown, and
`run_encoding` / `run_permutation` now record `n_repeats` in meta so it is
available. Results saved before this have no `n_repeats`; `--normalize` warns
and falls back rather than silently using the wrong ceiling.

**Partly addressed since (2026-09-09).** `--max-repeats 5` is now the default on
all four entry points (`02e0d3e`), so every subject's mask comes from the same
number of repeats and the *estimator variance* is matched. It did not make the
masks equal in size — at 5 repeats each, UTS01-03 are 6,555 / 11,254 / 16,395
voxels against 1,706-4,092 for the other six. That residual gap is a real
difference in data quality between the three dense subjects and the rest, not an
artefact of the repeat count, so "mean r over EV>0.1 voxels" is still not quite
the same quantity across subjects. Within a subject it cancels — every layer,
model and condition shares one mask — which is why the sweeps are safe and only
group-level statements are affected.

For group maps the common space is fsaverage, not native space:
`project_masked` writes NaN outside each subject's own mask and
`--min-subjects N` (default 5) drops vertices too few subjects reach.
`n_subjects_*.npy` is saved unthresholded, so the floor can be changed and the
averaging re-run without reprojecting. Whether thin coverage is also *biased
upward* — the subjects that reach a vertex being the ones whose EV mask included
it — is untested; bin mean r by `n_subjects` once the surface maps exist. The
variance argument for a floor holds either way.

### Lower priority

- **FIXED (warning added):** `run_encoding --backend huth --eval cv` is
  optimistically biased — `ridge_cv` picks alphas by LOO over all training
  stories and reuses them inside the CV it reports, while `fit_banded_cv`
  re-searches inside each outer fold. Latent, since every published number is
  banded and the documented `--backend both` recipes are all `--eval holdout`,
  but this file calls huth a "conservative lower bound" and on cv that is
  backwards.
- **FIXED for real (`6f8e28d`):** `--min-ev` used to be a silent no-op under
  `run_encoding --eval cv` — EV was computed only on the holdout path, while
  both sweeps honour it, so those cv numbers were over ~81,126 voxels and the
  sweeps' over ~1,776, with nothing saying so. EV is a function of Y alone, so
  it is now computed whenever the repeated story exists. The repeated story is
  still never in `train_stories`, so only the mask is taken and the fit is
  unchanged; a subject without repeats warns instead of silently scoring
  whole-brain.
- **FIXED:** `scripts/run_pipeline.sh` no longer passes the removed
  `--use-brain-pca` / `--brain-weight` flags.
- **NOT fixed:** `run_permutation` FDR-corrects over the EV ROI and then saves
  full-brain maps padded with `p=1`. Correct as saved, but it invites a wrong
  second correction downstream. Worth a note in whatever reads them.

### Fixed in d713d63

Three bugs in the new permutation code, all of which good-looking output would
have hidden: observed and null came from different estimators; the alpha search
used leave-one-story-out where production used `--n-splits 5`; and the old
prediction-shuffle null scored 290 TRs against an observed 291. See that commit.

## Story lists: use the derived intersection, not the shipped file

`data/derivative/common_stories_25.json` is unusable. Its participant keys are
`sub-UTS01` where `stories_for_subject` looks up `UTS01`, so it raises KeyError
on every subject — a 45-task array once reported COMPLETED while doing nothing.
Its list is also not the intersection: it holds `life` (UTS04 never heard it)
and `fromboyhoodtofatherhood` (UTS09 never heard it) while omitting `legacy` and
`thatthingonmyarm`, which all nine did.

`python scripts/make_common_stories.py` derives the truth from `all_stories.json`
and writes `common_stories_all9.json`: **25 stories shared by all nine,
`wheretheressmoke` among them, so 24 training stories.**

Per-subject lists (26-84 stories) are right for within-subject contrasts, where
training-set size cancels, and wrong for anything averaged across subjects.
They are also what made the all-stories run OOM: UTS01-03 at 27,797 training TRs
need >48 GB of VRAM, and even 80 GB was marginal.

**OpenNeuro v4.0.0 (2026-08-13) adds nothing here.** Checked: story counts per
subject are identical to ours and the 9-way intersection is the same 25. It adds
a 10th subject, UTS10, with 52 stories — but UTS10 **never heard
`wheretheressmoke`**, so there is no held-out story and no explainable-variance
ceiling for them, and including them would cut the intersection to 23.

Note an imbalance the common-story set does *not* fix: UTS01-03 have 10 repeats
of the held-out story, the other six have 5, so the EV mask and the holdout
noise floor are estimated about twice as precisely for those three.

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

## Running an array this wide: three traps

- **himalaya's progress bar kills array jobs.** It redraws stdout thousands of
  times per fit; under SLURM stdout is a file, and 36 concurrent tasks got
  `OSError: [Errno 121] Remote I/O error`. `default_solver_params` now passes
  `progress_bar=False`.
- **`/home` is a BeeGFS volume at 97% full and refuses log writes at that rate.**
  Symlink `logs/` to scratch before submitting. The failures are worse than
  cosmetic: tasks that had already written `sweep.csv` were marked FAILED
  because the closing `echo` could not reach the log, so "rerun the failures"
  redoes finished work. **Check the output files, not the SLURM exit states.**
- **A subject directory is not a finished fit, and counting them is not a
  completeness check.** `run_encoding` creates
  `banded/holdout/<SUBJ>/` when it *starts* that subject, so
  `ls banded/holdout | wc -l` returns 9 while five fits are still running. A
  permutation launched on that count skipped five of nine subjects
  (2026-09-21, k4 marginal): `load_observed_fit` raised
  `FileNotFoundError` on the missing `meta.json`, `run_subject` caught it per
  subject and logged "skipped", the array exited 0, and **sacct reported
  COMPLETED for every task** — three of them in 23 to 35 seconds. Count
  `*/meta.json` for an encoding run and `*/report.json` for a permutation run,
  or chain with `--dependency=afterok:` and never count at all.

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
- *Text context length.* DONE, 2026-09-01 — see the context-length section
  above. k=16 words, +0.031 in 9/9 subjects, the largest effect measured here.
- *Audio context window.* They feed 60 s chunks; we mean-pool a 2 s window per
  TR. The 2 s matches the eGeMAPS target windows, which was right for
  fine-tuning, but for *extraction* a longer window gives the transformer real
  context. Testable with the existing code. Still open.
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
- `scripts/` holds the things that are *run*, in three drawers (2026-09-21):

      scripts/                 one-off analysis utilities, run locally
        group_maps.py            group means + p-values from the nulls
        project_to_fsaverage.py  subject volume -> fsaverage
        avd_contrast.py  split_avd_bands.py  summarise_sweep.py
        make_common_stories.py  check_consistency.py  bench_solvers.py
        run_pipeline.sh          end-to-end driver (assumes scripts/, one
                                 level under the root -- do not move it)
      scripts/cluster/         every *.sbatch, plus setup_cluster.sh
      scripts/visualisation/   everything that turns a group directory into
                               something to look at or hand to another tool

  The visualisation scripts import `plot_group_maps.py` for colormaps, labels
  and default tests, so they sit in one directory on purpose; they reach the
  project root two levels up, and `plot_group_maps.py` loads
  `project_to_fsaverage.py` from the parent. Moving any of the three drawers
  needs those three path assumptions checked.
