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

## Stage-2 result: every layer, common stories (2026-08-29)

Supersedes the 2026-08-28 coarse sweep (every third layer, per-subject story
lists, 40-story cap). That one was directionally right about the emotion model
and wrong or blind about everything else, so read this section, not it.

Design: 9 subjects x 4 stores = 36 GPU tasks, **all nine on the same 24 training
stories** (`common_stories_all9.json`, the true intersection), every stored layer
plus averaged ranges, `--min-ev 0.1`, both `--eval cv` and `--eval holdout`.
Summarise with `python scripts/summarise_sweep.py --eval cv --tidy out.csv`.

**Choose on cv, report on holdout.** The held-out story is 291 TRs; its standard
errors run 0.0015-0.0058 against 0.0003-0.0014 cross-validated. It cannot
resolve a 0.005 effect, and it shows the emotion damage below as noise. Picking
the best row out of a holdout sweep is also selection on the test set.

**Fine-tuning cut both ways** (cv, paired per subject, Δ = ft − base):

    emotion  L6  -0.0001  4/9 worse   <- freeze boundary, identical weights
             L8  -0.0024  7/9   *
             L10 -0.0047  8/9   *
             L11 -0.0066  8/9   *
    robust   L17 -0.0012  7/9   *
             L20 +0.0042  0/9   *     <- ft BETTER in every subject
             L21 +0.0037  0/9   *
             L22 +0.0024  2/9   *

One reading fits both: the eGeMAPS objective pulls top layers toward acoustics.
Where they already carried brain-relevant prosody (emotion pretraining) that is
a loss; where they had specialised away from acoustics entirely (top of a
self-supervised speech stack) it is a partial recovery. Neither beats the frozen
emotion model.

**Two things the coarse sweep got wrong:**
- `base_emotion` does **not** keep climbing to the top of the stack. It rises to
  L10 (+0.0103) and plateaus at L11 (+0.0102). The peak is real, not truncation.
- `base_robust` has a **cliff, not a slope**: ~+0.007 through L19, then +0.0025
  at L20. Sampling L18 then L21 bracketed it without locating it.

**Averaged ranges never beat the best single layer.** emotion 9-11 +0.0098 vs
L10 +0.0103; robust 15-18 +0.0077 vs L17 +0.0080. Do not pay 4x the columns.

**Carry `base_emotion` L10 forward** — best layer of any model on cv, above
openSMILE in 9/9 subjects, and no fine-tuned checkpoint needed. L11 is
statistically indistinguishable.

Scale: openSMILE ~0.011 mean over the 1,776 of 81,126 voxels passing EV>0.1;
the best layer adds ~0.010. Consistent across all nine subjects, still small.

Published summary (both evaluations, every layer):
https://claude.ai/code/artifact/1b2b34ce-2eb3-447f-9809-35d5bbd4f39d

## Stage-2 result: semantic context length (2026-09-01)

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

**Context helps in 9/9 subjects, and it is the biggest effect in the project
so far** — +0.031 cv against the ~+0.010 the best audio layer buys over
openSMILE. k=16 and k=256 are indistinguishable; the dip at k=64 is noise.
Choosing on cv: **k=16**, same score for a sixteenth of the context.

## Stage-2 result: GPT-2 depth at k=16 (2026-09-08)

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
hourglass, measured. `--layers last` cost +0.0087 mean r, which is about what
the *entire* audio-layer effect buys over openSMILE (~+0.010).

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

## The final joint model: which bands, and one mismatch to fix first

The joint model takes the best-performing prosody band and the best-performing
semantic band, each chosen on **cv** over its own sweep. As of 2026-09-08 that is:

    prosody    perlayer_base_emotion  layer 10     +0.0103 over openSMILE, 9/9
    semantic   perlayer_gpt2_k16      layer 8      +0.0397 over gpt2_mean, 9/9

**The prosody choice does not match what the 45 held-out runs actually used.**
Those ran `base_emotion_L11` and `base_robust_L18`, because
`encoding_holdout.sbatch` was written against the *coarse* sweep, which sampled
every third layer and never tested L10 or L17. The corrected all-layer sweep
then found L10 (+0.0103 vs L11 +0.0102) and L17 (+0.0080 vs L18 +0.0078).
Numerically this is nothing — L10 and L11 are statistically indistinguishable.
As a *selection rule* it is not nothing: "we took the best layer on cv" has to
name L10, or the sweep table in the paper contradicts the methods section.
Decide one of:
  (a) use L10 and say so — costs one extraction, matches the stated rule;
  (b) keep L11 and state plainly that L10/L11 are within noise and L11 was
      already extracted. Defensible, but must be written down, not silent.

**There is no flat `base_emotion_L10` store**, only `base_emotion_L11`,
`base_emotion_9to11`, `base_robust_L18`, `base_robust_15to18`. Rather than
extract another flat store per layer, give `run_encoding` the `store:layer`
source syntax `run_semantic_sweep` already has (`perlayer_base_emotion:10`,
`perlayer_gpt2_k16:8`). One mechanism, no duplicated stores, and it removes the
class of error that produced this mismatch.

**Selection budgets are unequal, and that biases `preference` on cv.** The
audio band was chosen over ~96 configurations (4 stores x ~24 layers); the text
band over ~19 (13 layers at k=16, 7 context lengths at L12). Taking a max over
more candidates carries more winner's curse, so the *cross-validated* audio
score is the more optimistic of the two and `preference = r_text − r_audio` is
biased toward audio if read off cv. The held-out story is unaffected, because
neither selection touched it — one more reason to report `preference` on
holdout only, and to say in the methods how many configurations each band was
selected over.

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

**NOT fixed — and it cannot be fixed in code.** The mask instability at n=5 is a
property of the data, not a bug. Options are to report per-subject rather than
pooled, to fix one common mask (e.g. voxels passing EV in all nine, or a mask
built from a fixed 5 repeats for every subject so the estimator variance
matches), or to state the imbalance. Decide before the group maps.

### Lower priority

- **FIXED (warning added):** `run_encoding --backend huth --eval cv` is
  optimistically biased — `ridge_cv` picks alphas by LOO over all training
  stories and reuses them inside the CV it reports, while `fit_banded_cv`
  re-searches inside each outer fold. Latent, since every published number is
  banded and the documented `--backend both` recipes are all `--eval holdout`,
  but this file calls huth a "conservative lower bound" and on cv that is
  backwards.
- **FIXED (warning added):** `--min-ev` is a silent no-op under
  `run_encoding --eval cv` (EV needs the repeated story, which only the holdout
  path loads) while both sweeps honour it — so those cv numbers are over
  ~81,126 voxels and the sweeps' over ~1,776. Not comparable, and nothing said so.
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
