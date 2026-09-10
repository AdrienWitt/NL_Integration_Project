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

Rebuilt 2026-09-09 (`scripts/make_targets.sbatch`, 44 min, CPU only) and checked
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
nothing new. `scripts/finetune.sbatch` takes `SAVE_LIMIT` as an env var if a
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

    band                 dim    columns   mean r (EV>0.1, cv)
    emotion_avd            3        12    running
    opensmile             88       352    0.0867
    base_emotion L11    1024      4096    0.1504

Do **not** use `--output both`. The 3 values are a function of the 1024-d
vector, so concatenating returns an opaque band and throws away the only thing
worth having: three columns with names.

**Dominance is not a separate dimension.** Measured on the 774 stimuli of
`../Clean_Irony/embeddings/audio_wav2vec_avd/` — this exact model's outputs:

    arousal x dominance   0.950      variance per component:
    arousal x valence     0.223          0.639 / 0.348 / 0.013
    dominance x valence   0.274      -> effective rank 2, not 3

Likely a property of the model (MSP-Podcast dominance annotations track arousal)
rather than of that corpus; `scripts/split_avd_bands.py` reprints the matrix on
the LeBel stimuli so it can be confirmed here. Consequences:

- **Valence vs arousal (r ~ 0.25) is the identifiable contrast**, and the one
  with a literature behind it. Report it as primary.
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

**`run_encoding` now takes N named bands** (`6f8e28d`): `--band NAME=STORE`,
repeatable, replacing the text/audio pair, because a decomposition *inside* one
modality is three bands of one modality rather than a text band and an audio
band. Each band still gets its own alpha; `split_corrs` still reports each
band's share of the joint prediction — shares that sum to the joint r, **not**
standalone correlations. Without `--band` nothing changes: the two-band design
matrix is column-for-column identical, and `MODEL_BANDS` keeps its name because
`stats.run_permutation` imports it.

Note `--n-iter 200` rather than the sweeps' 20: random search samples the
simplex of band weights, 20 was calibrated for p=4096 where each fit is
expensive, and here the band weights are exactly what two collinear dimensions
are fighting over.

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

## Stage-2 result: GPT-2 depth at k=16 (2026-09-08)

> **On the old EV masks**, and being re-run on the current ones
> (job submitted 2026-09-09). The L8 argmax should survive — mask changes
> affect all layers alike — but the absolute r will move for UTS01-03.

The context sweep above ran entirely at `--layers last`, which on a *causal*
LM is the layer least likely to win: the top of the stack is optimised to emit
next-token logits and rotates back toward the unembedding matrix, away from
the abstract middle (logit lens, nostalgebraist 2020; tuned lens, Belrose et
al. 2023). So +0.031 was measured at a probable trough. It was.

9 subjects x 13 hidden states x both evals, k=16, `common_stories_all9`,
`--min-ev 0.1`. Mean r over the EV>0.1 voxels, averaged over subjects:

    layer     cv      vs L12   n     holdout   vs L12   n
    L0      0.1476   -0.0181  0/9    0.2914   -0.0357  0/9
    L1      0.1523   -0.0134  0/9    0.3031   -0.0239  1/9
    L2      0.1551   -0.0106  0/9    0.3049   -0.0222  1/9
    L3      0.1587   -0.0070  1/9    0.3103   -0.0168  3/9
    L4      0.1622   -0.0035  1/9    0.3178   -0.0093  3/9
    L5      0.1658   +0.0002  4/9    0.3236   -0.0035  5/9
    L6      0.1694   +0.0038  9/9    0.3281   +0.0011  5/9
    L7      0.1727   +0.0071  9/9    0.3348   +0.0078  7/9
    L8      0.1744   +0.0087  9/9    0.3394   +0.0123  8/9   <- peak, both evals
    L9      0.1734   +0.0077  9/9    0.3363   +0.0092  8/9
    L10     0.1713   +0.0057  9/9    0.3364   +0.0093  9/9
    L11     0.1682   +0.0025  9/9    0.3328   +0.0058  8/9
    L12     0.1657    —       0/9    0.3271    —       0/9   <- what the k sweep used

**A clean inverted U peaking at L8 of 12 — two thirds of the way up.** The
hourglass, measured. `--layers last` cost +0.0087 mean r. (This used to add
"about what the entire audio-layer effect buys over openSMILE (~+0.010)" — that
was the misaligned audio band; the corrected figure is +0.065, so depth on the
text side is a seventh of it, not a match for it.)

**L8 is the argmax in all nine subjects independently** on cv — not a group
mean with a soft peak, nine separate maxima at the same layer. Holdout (all
nine, complete) puts the peak at L8 in five subjects, L10 in two, L9 and L11
in one each: the same shape, resolved less sharply, exactly as its ~4x larger
standard errors predict. Read L8-L10 as a plateau there, and take the choice
from cv as the rule says.

**The peak is not near the bottom**, which settles the interpretive worry that
motivated the sweep: at two thirds depth the band is not a lookup table, and
calling it semantic in `preference = r_text − r_audio` is defensible. Report
the profile alongside the argmax anyway — the shape is the evidence.

**Carry `perlayer_gpt2_k16:8` forward as the semantic band.** Against the
original `gpt2_mean` it is +0.0397 cv (context +0.0310, depth another +0.0087),
9/9 subjects.

Consistency check, built into the run: `gpt2_k16` (the flat store the context
sweep used) and `perlayer_gpt2_k16:12` are the same layer from the same
forward passes, and their encoding scores agree to **0.00e+00 in every
subject**. The per-layer path is verified end to end, not just at the feature
level.

Read it with:

    python scripts/summarise_sweep.py --root results/encoding/semantic_sweep/cv \
        --baseline gpt2_mean

## Still open on the semantic band: k at L8

Depth and context can interact, and the k sweep was run at L12, the one layer
least likely to reveal it. Re-check k=4/16/256 at L8 before freezing the band.
This needs new per-layer extractions at those k (only k=16 has one), so it is
~20 min of GPU per k plus a sweep:

    sbatch --array=0-1 --export=ALL,KS="4 256",LAYERS="0-12 0-12",PER_LAYER=1,\
           OUTNAMES="perlayer_gpt2_k4 perlayer_gpt2_k256" \
           scripts/extract_semantic.sbatch
    sbatch --time=08:00:00 --dependency=afterok:<jobid> \
           --export=ALL,SOURCES="perlayer_gpt2_k4:8 perlayer_gpt2_k16:8 perlayer_gpt2_k256:8",\
           TAGSUF=_kAtL8 scripts/semantic_sweep.sbatch

`gpt2_mean` is deliberately absent from SOURCES: the sbatch always passes it
as `--baseline-features`, and naming it in both scores the same band twice.

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
    semantic   perlayer_gpt2_k16 layer 8    +0.0397 over gpt2_mean, 9/9
                                      (being re-scored on the current masks)

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

**The semantic numbers are on the old EV masks.** Every semantic sweep predates
`--max-repeats 5` (2026-09-09), so UTS01-03 were scored over masks built from 10
repeats while every prosody number above comes from 5 — 1,776 voxels against
6,555 for UTS01. A depth sweep at k=16 on the current masks is running; the old
results are archived at `results/encoding/semantic_sweep_oldmask_20260909/`.
**Until it lands, do not put a text number and an audio number side by side.**

**Selection budgets are unequal, and that biases `preference` on cv.** The audio
band was chosen over ~96 configurations (4 stores x ~24 layers), the text band
over ~19. A max over more candidates carries more winner's curse, so the
cross-validated audio score is the more optimistic of the two. The held-out story
is unaffected, since neither selection touched it — one more reason to report
`preference` on holdout only, and to say in the methods how many configurations
each band was selected over.

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
