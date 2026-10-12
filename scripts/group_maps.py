#!/usr/bin/env python3
"""
Group maps with per-vertex p-values, on fsaverage.

    per-subject permutation output  --project--> fsaverage
                                    --combine--> group mean + p

`scripts/project_to_fsaverage.py` already averages subject maps; what it cannot
do is put a p-value on the average, because `write_group_mean` only ever sees
the observed maps. This script reads the *nulls* as well (run the permutation
with `--save-null`) and produces three group tests from them. They answer
different questions and are reported side by side rather than one being picked:

1.  ``p_perm`` -- group permutation, using the fact that the nine subjects share
    a seed. `run_permutation` draws its block shuffles from
    `default_rng(--seed)` over `n_train` TRs, and all nine subjects have the
    same 24 common stories, hence the same 8,683 training TRs. So null draw *i*
    is the *identical* shuffle of time for every subject, the per-draw group
    mean is a valid null for the observed group mean, and the p-value has
    1/(n_perms+1) resolution. This is a fixed-effects statement: it is about
    these nine people and this stimulus, not a population.

2.  ``p_signflip_fwe`` -- max-statistic sign-flip FWE over the 2^9 = 512 sign
    vectors (Winkler et al. 2014, PALM). Random effects, so it does generalise,
    but it is coarse: every p is a multiple of 1/512 and the floor is 0.00195.
    That floor is fatal for BH and harmless here, because the correction lives
    in the null of the *max* -- the threshold sits at rank 486/512 and needs no
    per-vertex p below alpha/n.

    Sign-flipping assumes the subject statistic is symmetric about 0 under H0,
    and a raw `Delta r` is not: adding a shuffled band's columns to a ridge
    moves r by a small amount that is not centred. So each subject's map is
    centred on its own permutation null mean before flipping. This is the whole
    reason the nulls have to be kept.

3.  ``p_prevalence`` -- a binomial count (Allefeld, Goergen & Haynes 2016).
    Under H0 a permutation p-value is uniform, so the number of subjects
    significant at an uncorrected alpha is exactly Binomial(n_v, alpha). Tests
    "more than alpha of the population has an effect here", which is a weaker
    claim in kind than a mean test but the one nine subjects really support,
    and it reaches p = 6e-04 at 4 of 9 -- finer resolution than sign-flipping.

    The per-subject p is recomputed *on the surface* from the projected null,
    never by projecting the volume p-map: interpolating p-values is a smoothing
    operation and the result is no longer uniform under H0. Projection is a
    fixed linear map applied identically to observed and null, so recomputing
    after it preserves exchangeability vertex by vertex.

4.  ``p_tfce_fwe`` -- the same sign-flip over TFCE of the moderated t
    (Smith & Nichols 2009; E=1, H=2, FreeSurfer's average vertex area), run
    afterwards from the saved stacks with ``--tfce``. Random effects like (2),
    but it borrows strength from neighbouring vertices, so it can see weak
    extended effects that (2) only sees in the n>=9 core. Its family is n>=7
    and a subject missing at a family vertex is filled with 0, its own null
    mean -- conservative and exact. Significance means "part of a significant
    region", not a localised vertex: localisation stays with (1)-(3) and the
    parcel tables. Vertices in small isolated patches of the family cannot
    borrow strength and are systematically missed.

Coverage
--------
A vertex is not reached by every subject: the slab does not cover all cortex,
and `--min-ev` leaves most voxels unfitted. Every statistic is therefore
computed over the subjects that actually reach a vertex, with the count saved
as `n_subjects.npy`, and vertices below `--min-subjects` are NaN-ed out of the
saved maps. `p_perm` and `p_prevalence` handle a variable *n* exactly (the
observed and null group means are taken over the same subject set; the binomial
uses the per-vertex n). The sign-flip max-null is the one that does not: it
compares a t across vertices backed by different numbers of subjects. t is
pivotal, which is why PALM does the same thing, but read it as the conservative
map of the three.

Conjunction
-----------
For a pair of conditional contributions the integration statistic is the
intersection-union test, so its p is the per-vertex `max` of the two component
p-values (Berger 1982) -- computed separately within each of the three
families, and with no correction between the two components.

Usage
-----
    python scripts/group_maps.py \
        --perm-dir results/encoding/<run>__shuffle_conditional__null/permutation \
        --out results/group/<run>_conditional
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np

log = logging.getLogger("group_maps")

_HERE = Path(__file__).resolve().parent


def _load_projector_module():
    """`project_to_fsaverage` is a script, not a module; load it by path."""
    spec = importlib.util.spec_from_file_location(
        "project_to_fsaverage", _HERE / "project_to_fsaverage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Statistics to look for, and the conjunctions to build from them.
CONJUNCTIONS = {
    "delta": ("delta_audio_given_text", "delta_text_given_audio"),
}

# Differences between two single-band models. Unlike a conjunction these cannot
# reuse the component p-values: the prediction-shuffle null behind `text` and
# `audio` is H0 "this model does not predict at all", under which BOTH r go to
# ~0 and so does their difference -- it tests for signal, not for one band
# beating the other. The null wanted is "the two bands predict equally well",
# and across subjects that is a paired test on the per-subject difference.
CONTRASTS = {
    "preference": ("text", "audio"),
    # In the arousal+valence model the joint term cancels exactly, as for
    # preference: (r_av - r_a) - (r_av - r_v) = r_v - r_a. So this is the
    # unique-contribution contrast with dominance -- the arousal twin at
    # r = 0.963 -- left out, not a comparison against the 3-band leftovers.
    "valence_vs_arousal": ("valence", "arousal"),
}


def discover_stats(subject_dir: Path) -> list[str]:
    """Statistic names that have both an observed map and a saved null."""
    names = []
    for obs in sorted(subject_dir.glob("*_observed.npy")):
        name = obs.name[: -len("_observed.npy")]
        if (subject_dir / f"{name}_null.npy").exists():
            names.append(name)
    return names


class SubjectMaps:
    """One subject's observed map and null stack, projected to fsaverage."""

    def __init__(self, projector, subject_dir: Path):
        self.projector = projector
        self.dir = subject_dir
        self.voxel_mask = np.load(subject_dir / "voxel_mask.npy").astype(bool)
        # The mask's own interpolation weight. project_masked recomputes this
        # for every call; it is subject-constant, so it is computed once here
        # and the 1,000 null projections cost one _project_raw each instead of
        # two.
        self.weights = projector._project_raw(
            self.voxel_mask.astype(np.float32))
        self.reached = self.weights > 0

    def _project_masked(self, full_volume: np.ndarray) -> np.ndarray:
        filled = np.nan_to_num(np.asarray(full_volume, dtype=np.float64),
                               nan=0.0)
        filled[~self.voxel_mask] = 0.0
        num = self.projector._project_raw(filled.astype(np.float32))
        out = np.full(self.weights.shape, np.nan, dtype=np.float32)
        out[self.reached] = num[self.reached] / self.weights[self.reached]
        return out

    def observed(self, name: str) -> np.ndarray:
        return self._project_masked(np.load(self.dir / f"{name}_observed.npy"))

    def null(self, name: str) -> np.ndarray:
        """(n_perms, n_vertices). Nulls are saved in mask space."""
        null = np.load(self.dir / f"{name}_null.npy")
        n_perms = null.shape[0]
        out = np.empty((n_perms, self.weights.size), dtype=np.float32)
        scratch = np.zeros(self.voxel_mask.size, dtype=np.float64)
        for i in range(n_perms):
            scratch[:] = 0.0
            scratch[self.voxel_mask] = null[i]
            out[i] = self._project_masked(scratch)
            if (i + 1) % 200 == 0:
                log.info(f"      projected {i + 1}/{n_perms} null draws")
        return out


def sign_flip_fwe(centred: np.ndarray, valid: np.ndarray,
                  var_floor_pct: float = 25.0) -> np.ndarray:
    """Max-statistic sign-flip FWE p over every 2^n sign vector.

    `centred` is (n_subjects, n_vertices) with NaN where a subject does not
    reach a vertex; `valid` selects the vertices the family is tested over.
    Returns a full-length p-map, NaN outside `valid`.

    The statistic is a *moderated* one-sample t, and the moderation is not
    cosmetic. A plain max-t is set by whichever vertex has the smallest
    dispersion, not by whichever has the largest effect: nine subjects who
    happen to agree closely give t -> infinity as sd -> 0, and a handful of such
    vertices anywhere in the map raise the null maximum above every real effect.
    Measured here on the first run: max-t put the smallest attainable FWE p at
    0.047, so nothing survived at all, while a simulation with a known effect
    reached t = 14.5 where the effect was and t = 80.4 at no-effect vertices
    whose variance had collapsed.

    So the denominator is floored at the `var_floor_pct`-th percentile of the
    observed sd. The floor is a single constant taken once from the unflipped
    data and applied identically to every flip, which leaves the test exact --
    it is a different statistic, not a corrected p.
    """
    n_subj = centred.shape[0]
    data = centred[:, valid]

    sd_obs = np.nanstd(data, axis=0, ddof=1)
    finite = np.isfinite(sd_obs) & (sd_obs > 0)
    sd_floor = (np.percentile(sd_obs[finite], var_floor_pct)
                if finite.any() else 0.0)
    log.info(f"      sd floor at p{var_floor_pct:g} = {sd_floor:.5f} "
             f"({int((sd_obs < sd_floor).sum()):,} vertices moderated)")

    def tstat(signed: np.ndarray) -> np.ndarray:
        n = np.sum(~np.isnan(signed), axis=0)
        mean = np.nanmean(signed, axis=0)
        # ddof=1; a vertex backed by one subject has no dispersion estimate.
        sd = np.maximum(np.nanstd(signed, axis=0, ddof=1), sd_floor)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = mean / (sd / np.sqrt(np.maximum(n, 1)))
        return np.where((n >= 2) & (sd > 0), t, np.nan)

    observed_t = tstat(data)
    maxima = np.empty(2 ** n_subj, dtype=np.float64)
    for b, signs in enumerate(product((1.0, -1.0), repeat=n_subj)):
        t = tstat(data * np.asarray(signs, dtype=np.float32)[:, None])
        maxima[b] = np.nanmax(t) if np.any(~np.isnan(t)) else -np.inf

    # The identity flip is one of the 2^n, so the observed map is already in
    # the null and no +1 correction is needed.
    p = np.array([np.mean(maxima >= t) if np.isfinite(t) else np.nan
                  for t in observed_t])
    out = np.full(centred.shape[1], np.nan, dtype=np.float64)
    out[valid] = p
    return out


def fsaverage_mesh():
    """Vertex adjacency and vertex area over both hemispheres, lh-then-rh.

    The same order `project_to_fsaverage.Projector.to_fsaverage` concatenates
    in, asserted against the surface. Area is FreeSurfer's
    `?h.white.avg.area.mgh` -- each vertex's white-surface area averaged over
    the subjects fsaverage was built from -- which is what PALM and
    FreeSurfer's own TFCE use on fsaverage, where the template's triangles are
    far from uniform and a vertex count would weight dense regions twice.
    """
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    from scipy import sparse

    sys.path.insert(0, str(_HERE / "visualisation"))
    from plot_group_inflated import fsaverage_dir
    fsdir = fsaverage_dir()

    rows, cols, areas, offset = [], [], [], 0
    for hemi in ("lh", "rh"):
        coords, faces = fsio.read_geometry(fsdir / "surf" / f"{hemi}.white")
        area = np.asarray(nib.load(fsdir / "surf" /
                                   f"{hemi}.white.avg.area.mgh").get_fdata()
                          ).ravel()
        if area.size != coords.shape[0]:
            raise SystemExit(f"{hemi}: {area.size} areas for "
                             f"{coords.shape[0]} vertices")
        for a, b in ((0, 1), (1, 2), (2, 0)):
            rows.append(faces[:, a] + offset)
            cols.append(faces[:, b] + offset)
        areas.append(area)
        offset += coords.shape[0]
    r, c = np.concatenate(rows), np.concatenate(cols)
    adj = sparse.coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)),
                            shape=(offset, offset)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.int8)
    return adj, np.concatenate(areas)


def tfce(values: np.ndarray, adj, area: np.ndarray, dh: float,
         E: float = 1.0, H: float = 2.0) -> np.ndarray:
    """Threshold-free cluster enhancement of the positive tail (Smith &
    Nichols 2009): TFCE(v) = sum_h e(h, v)^E * h^H * dh, with e the area of the
    suprathreshold component containing v at height h.

    E=1, H=2 are the surface defaults (FSL/PALM); the volumetric E=0.5 assumes
    three-dimensional clusters and under-weights extent on a 2-d sheet.
    `adj` and `area` must already be restricted to the vertices of `values`.
    """
    from scipy.sparse.csgraph import connected_components

    out = np.zeros(values.size, dtype=np.float64)
    top = np.nanmax(values) if values.size else 0.0
    h = dh
    while h <= top:
        above = np.flatnonzero(values >= h)
        _, labels = connected_components(adj[above][:, above], directed=False)
        extent = np.bincount(labels, weights=area[above])
        out[above] += extent[labels] ** E * h ** H * dh
        h += dh
    return out


def sign_flip_tfce(centred: np.ndarray, valid: np.ndarray, adj,
                   area: np.ndarray, var_floor_pct: float = 25.0,
                   n_steps: int = 50, E: float = 1.0, H: float = 2.0):
    """FWE p on TFCE of the moderated t, max-statistic over every sign vector.

    Returns (p, tfce_observed), full length, NaN outside `valid`.

    Unlike `sign_flip_fwe` this cannot let *n* vary by vertex: TFCE integrates
    t over neighbours, so a vertex backed by five subjects would lend its
    noisier t to the cluster around it. A subject missing at a family vertex is
    therefore filled with 0 -- in a stack centred on each subject's own null
    mean, exactly "this subject contributes the null here". That is
    conservative (it pulls the mean toward 0 and widens the sd) and it keeps
    the test exact, because the fill is the same under every flip.

    The t is the same moderated one as `sign_flip_fwe`, for the same reason: an
    unmoderated t explodes where nine subjects happen to agree, and TFCE would
    then build clusters out of collapsed variance.
    """
    n_subj = centred.shape[0]
    data = np.nan_to_num(centred[:, valid], nan=0.0).astype(np.float64)
    sub_adj = adj[valid][:, valid]
    sub_area = area[valid]
    n_filled = int(np.isnan(centred[:, valid]).sum())
    log.info(f"      {int(valid.sum()):,} vertices, {n_filled:,} missing "
             f"subject-vertex values filled with the null (0)")

    sd_obs = data.std(axis=0, ddof=1)
    sd_floor = np.percentile(sd_obs[sd_obs > 0], var_floor_pct)

    def tstat(signed):
        sd = np.maximum(signed.std(axis=0, ddof=1), sd_floor)
        return signed.mean(axis=0) / (sd / np.sqrt(n_subj))

    t_obs = tstat(data)
    # One step size for every flip, fixed from the observed map, so each null
    # map is integrated on the same grid as the one it is compared with.
    dh = max(float(np.max(t_obs)), 1e-6) / n_steps
    tfce_obs = tfce(t_obs, sub_adj, sub_area, dh, E, H)

    maxima = np.empty(2 ** n_subj, dtype=np.float64)
    for b, signs in enumerate(product((1.0, -1.0), repeat=n_subj)):
        t = tstat(data * np.asarray(signs)[:, None])
        maxima[b] = tfce(t, sub_adj, sub_area, dh, E, H).max()
        if b and b % 128 == 0:
            log.info(f"      {b}/{2 ** n_subj} flips")

    # The identity flip is in the null, as in sign_flip_fwe.
    p = (maxima[None, :] >= tfce_obs[:, None]).mean(axis=1)
    p_full = np.full(centred.shape[1], np.nan)
    t_full = np.full(centred.shape[1], np.nan)
    p_full[valid], t_full[valid] = p, tfce_obs
    return p_full, t_full


def run_tfce(out_dir: Path, alpha: float, tfce_min_subjects: int,
             var_floor_pct: float, n_steps: int, E: float, H: float) -> None:
    """TFCE sign-flip FWE for every statistic in a group directory, from the
    saved centred stacks; no projection. Conjunctions take the per-vertex max
    of their components' p, as for every other family."""
    stacks = sorted(out_dir.glob("*__centred_stack.npy"))
    if not stacks:
        raise SystemExit(f"no *__centred_stack.npy under {out_dir}")
    adj, area = fsaverage_mesh()
    done = {}
    for path in stacks:
        name = path.name[: -len("__centred_stack.npy")]
        centred = np.load(path)
        if centred.shape[1] != area.size:
            raise SystemExit(f"{name}: {centred.shape[1]} vertices, mesh has "
                             f"{area.size}")
        count = np.load(out_dir / f"{name}__n_subjects.npy")
        family = count >= min(tfce_min_subjects, int(count.max()) or 1)
        log.info(f"  {name}: TFCE family n>={tfce_min_subjects}")
        p, enhanced = sign_flip_tfce(centred, family, adj, area,
                                     var_floor_pct, n_steps, E, H)
        np.save(out_dir / f"{name}__p_tfce_fwe.npy", p)
        np.save(out_dir / f"{name}__tfce.npy", enhanced.astype(np.float32))
        np.save(out_dir / f"{name}__tfce_family.npy", family)
        done[name] = p
        log.info(f"    p<{alpha}: {int(np.nansum(p < alpha)):,} of "
                 f"{int(family.sum()):,} vertices")

    for conj, (a, b) in CONJUNCTIONS.items():
        if a in done and b in done:
            p = np.fmax(done[a], done[b])
            np.save(out_dir / f"{conj}__p_tfce_fwe.npy", p)
            log.info(f"  {conj}: p<{alpha}: {int(np.nansum(p < alpha)):,} "
                     f"vertices")


def fdr(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """BH over the finite entries; NaN stays NaN."""
    from statsmodels.stats.multitest import fdrcorrection
    out = np.full(pvals.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(pvals)
    if finite.any():
        _, corrected = fdrcorrection(pvals[finite], alpha=alpha)
        out[finite] = corrected
    return out


def compute_contrasts(out_dir: Path, alpha: float, fwe_min_subjects: int,
                      var_floor_pct: float, min_subjects: int) -> None:
    """Between-model differences, from the saved centred stacks.

    `preference = r_text - r_audio` is one of this project's headline
    statistics and it has no permutation of its own, because no null in
    `run_permutation` models "the two bands are equally good". Two group tests
    are available without one, and both are paired, which is the right shape:
    the same subject contributes both numbers, so their difference removes
    everything about that subject's data quality.

    - sign-flip FWE on the per-subject difference (random effects, exact).
    - the exact sign test on how many subjects favour text, against
      Binomial(n_v, 0.5). This is the same 1/512-floor test the sweeps use for
      "9/9 subjects", applied per vertex.

    The stacks are already centred on each statistic's own null mean, so the
    difference is centred too and sign-flipping is applied to something
    symmetric about zero under H0.
    """
    from scipy.stats import binom

    for name, (a, b) in CONTRASTS.items():
        pa = out_dir / f"{a}__centred_stack.npy"
        pb = out_dir / f"{b}__centred_stack.npy"
        if not (pa.exists() and pb.exists()):
            continue
        diff = np.load(pa) - np.load(pb)
        count = np.load(out_dir / f"{a}__n_subjects.npy")
        family = count >= min(fwe_min_subjects, int(count.max()) or 1)
        log.info(f"  {name} = {a} - {b}: FWE family n>={fwe_min_subjects} "
                 f"({int(family.sum()):,} vertices)")

        # The same coverage floor the other maps use. Without it this map
        # alone would extend over single-subject vertices, and since mean r
        # rises monotonically with coverage those are both the noisiest and the
        # weakest -- a figure showing far more cortex than its neighbours for
        # no better reason than a different default.
        covered = count >= max(1, min_subjects)
        mean = np.where(covered, np.nanmean(diff, axis=0), np.nan)
        k = np.nansum(diff > 0, axis=0).astype(np.int16)
        n_v = np.maximum(count, 1)
        p_sign = np.where(covered,
                          binom.sf(np.maximum(k - 1, 0), n_v, 0.5), np.nan)
        p_fwe = sign_flip_fwe(diff.astype(np.float32), family,
                              var_floor_pct=var_floor_pct)

        np.save(out_dir / f"{name}__mean.npy", mean.astype(np.float32))
        np.save(out_dir / f"{name}__k_favouring_{a}.npy", k)
        np.save(out_dir / f"{name}__n_subjects.npy", count)
        np.save(out_dir / f"{name}__p_signtest.npy", p_sign)
        np.save(out_dir / f"{name}__p_signtest_fdr.npy", fdr(p_sign, alpha))
        np.save(out_dir / f"{name}__p_signflip_fwe.npy", p_fwe)
        np.save(out_dir / f"{name}__fwe_family.npy", family)

        cov5 = covered
        log.info(f"    mean {np.nanmean(mean[cov5]):+.4f} | "
                 f"{a} ahead at {np.nanmean(mean[cov5] > 0):.1%} of covered | "
                 f"sign test FDR {int(np.nansum(fdr(p_sign, alpha) < alpha)):,} | "
                 f"FWE {int(np.nansum(p_fwe < alpha)):,}")


def recompute_fwe(out_dir: Path, alpha: float, fwe_min_subjects: int,
                  var_floor_pct: float) -> None:
    """Redo only the sign-flip FWE, from the saved centred stacks.

    The stacks are what make this possible: the group mean and the permutation
    and prevalence p-values need the 9,000 projected null maps, but the
    sign-flip test needs only the nine centred subject maps. Retuning the FWE
    family or the variance floor is therefore seconds rather than the ~20 min
    of projection that produced the directory.
    """
    stacks = sorted(out_dir.glob("*__centred_stack.npy"))
    if not stacks:
        raise SystemExit(f"no *__centred_stack.npy under {out_dir}")
    done = {}
    for path in stacks:
        name = path.name[: -len("__centred_stack.npy")]
        centred = np.load(path)
        count = np.load(out_dir / f"{name}__n_subjects.npy")
        family = count >= min(fwe_min_subjects, int(count.max()) or 1)
        log.info(f"  {name}: FWE family n>={fwe_min_subjects} "
                 f"({int(family.sum()):,} vertices)")
        p_fwe = sign_flip_fwe(centred, family, var_floor_pct=var_floor_pct)
        np.save(out_dir / f"{name}__p_signflip_fwe.npy", p_fwe)
        np.save(out_dir / f"{name}__fwe_family.npy", family)
        done[name] = p_fwe
        log.info(f"    p<{alpha}: {int(np.nansum(p_fwe < alpha)):,} vertices")

    for conj, (a, b) in CONJUNCTIONS.items():
        if a in done and b in done:
            p = np.fmax(done[a], done[b])
            np.save(out_dir / f"{conj}__p_signflip_fwe.npy", p)
            log.info(f"  {conj}: p<{alpha}: {int(np.nansum(p < alpha)):,} "
                     f"vertices")


def run_statistic(name: str, subjects: dict, alpha: float,
                  min_subjects: int, fwe_min_subjects: int) -> dict:
    """Group mean and the three p-maps for one statistic."""
    from scipy.stats import binom

    n_vertices = next(iter(subjects.values())).weights.size
    obs_sum = np.zeros(n_vertices, dtype=np.float64)
    count = np.zeros(n_vertices, dtype=np.int16)
    null_sum = None
    per_subject_sig = []
    centred = []

    for subject, maps in subjects.items():
        log.info(f"    {subject}: projecting {name}")
        obs = maps.observed(name)
        null = maps.null(name)
        if null_sum is None:
            # float32: a (1000, 327684) float64 accumulator is 2.6 GB, and
            # the division below would allocate a second one.
            null_sum = np.zeros(null.shape, dtype=np.float32)

        reached = maps.reached
        obs_sum[reached] += obs[reached]
        null_sum[:, reached] += null[:, reached]
        count[reached] += 1

        # Per-subject p ON THE SURFACE, for the prevalence count.
        exceed = (null >= obs[None, :]).sum(axis=0)
        p_subject = (exceed + 1) / (null.shape[0] + 1)
        sig = np.where(reached, p_subject <= alpha, False)
        per_subject_sig.append(sig)

        # Centre on this subject's own null mean, so sign-flipping is applied
        # to something that is symmetric about zero under H0.
        d = np.full(n_vertices, np.nan, dtype=np.float32)
        d[reached] = obs[reached] - null[:, reached].mean(axis=0)
        centred.append(d)
        del null

    covered = count >= max(1, min_subjects)
    with np.errstate(invalid="ignore"):
        group_obs = np.where(count > 0, obs_sum / np.maximum(count, 1), np.nan)
        # In place: null_sum is the largest array here and is not needed after.
        null_sum /= np.maximum(count, 1)[None, :].astype(np.float32)
        group_null = null_sum

    n_perms = group_null.shape[0]
    exceed = (group_null >= group_obs[None, :]).sum(axis=0)
    p_perm = np.where(covered, (exceed + 1) / (n_perms + 1), np.nan)

    k = np.sum(per_subject_sig, axis=0)
    p_prev = np.where(covered & (count > 0),
                      binom.sf(np.maximum(k - 1, 0), np.maximum(count, 1),
                               alpha),
                      np.nan)
    p_prev = np.where(covered & (k == 0), 1.0, p_prev)

    centred = np.asarray(centred, dtype=np.float32)
    # The FWE family is deliberately NOT `covered`. A max-statistic is set by
    # the most extreme vertex anywhere in the family, and a vertex backed by
    # five subjects produces extreme flipped t far more readily than one backed
    # by nine -- so including the thinly-covered majority raises the null
    # maximum above every real effect. Measured on delta_audio_given_text:
    # 19 significant vertices with the family at n>=5 (61,168 vertices), 229 at
    # n>=7, and 2,279 of 3,337 at n>=9. The p-values are valid at every floor;
    # what changes is whether the test can see anything. n>=9 is also the only
    # homogeneous family, every vertex carrying the same degrees of freedom.
    fwe_family = count >= min(fwe_min_subjects, int(count.max()) or 1)
    p_fwe = sign_flip_fwe(centred, fwe_family)

    return {
        # 9 x n_vertices, ~12 MB. Saved so the group statistic can be changed
        # without re-projecting 9,000 null maps.
        "centred_stack": centred,
        "mean": np.where(covered, group_obs, np.nan).astype(np.float32),
        "mean_centred": np.where(covered, np.nanmean(centred, axis=0),
                                 np.nan).astype(np.float32),
        "n_subjects": count.astype(np.int16),
        "k_significant": k.astype(np.int16),
        "p_perm": p_perm,
        "p_perm_fdr": fdr(p_perm, alpha),
        "p_prevalence": p_prev,
        "p_prevalence_fdr": fdr(p_prev, alpha),
        "p_signflip_fwe": p_fwe,
        "fwe_family": fwe_family,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrasts", action="store_true",
                    help="compute between-model contrasts (preference) in "
                         "--out from the saved centred stacks")
    ap.add_argument("--recompute-fwe", action="store_true",
                    help="redo only the sign-flip FWE in --out, from the "
                         "saved centred stacks; no projection")
    ap.add_argument("--tfce", action="store_true",
                    help="TFCE sign-flip FWE in --out, from the saved centred "
                         "stacks; no projection")
    ap.add_argument("--tfce-min-subjects", type=int, default=7,
                    help="coverage floor for the TFCE family; missing "
                         "subjects inside it count as the null")
    ap.add_argument("--tfce-steps", type=int, default=50)
    ap.add_argument("--tfce-E", type=float, default=1.0)
    ap.add_argument("--tfce-H", type=float, default=2.0)
    ap.add_argument("--var-floor-pct", type=float, default=25.0)
    ap.add_argument("--perm-dir", required=False, type=Path,
                    help="a run's permutation/ directory (subject subdirs)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stats", nargs="*", default=None,
                    help="statistic names; default = every one with a null")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-subjects", type=int, default=5,
                    help="coverage floor for the mean, perm and prevalence maps")
    ap.add_argument("--fwe-min-subjects", type=int, default=9,
                    help="coverage floor for the sign-flip FWE family "
                         "(default 9: only vertices every subject reaches)")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.contrasts:
        compute_contrasts(args.out, args.alpha, args.fwe_min_subjects,
                          args.var_floor_pct, args.min_subjects)
        return 0
    if args.recompute_fwe:
        recompute_fwe(args.out, args.alpha, args.fwe_min_subjects,
                      args.var_floor_pct)
        return 0
    if args.tfce:
        run_tfce(args.out, args.alpha, args.tfce_min_subjects,
                 args.var_floor_pct, args.tfce_steps, args.tfce_E, args.tfce_H)
        return 0
    if args.perm_dir is None:
        raise SystemExit("--perm-dir is required unless --recompute-fwe")

    subject_dirs = sorted(d for d in args.perm_dir.iterdir()
                          if d.is_dir() and (d / "voxel_mask.npy").exists())
    if args.subjects:
        subject_dirs = [d for d in subject_dirs if d.name in args.subjects]
    if not subject_dirs:
        raise SystemExit(f"no subject directories under {args.perm_dir}")

    names = args.stats or discover_stats(subject_dirs[0])
    if not names:
        raise SystemExit(
            f"no *_null.npy under {subject_dirs[0]} -- rerun the permutation "
            f"with --save-null (SAVE_NULL=1 in scripts/cluster/permutation.sbatch)")
    log.info(f"{len(subject_dirs)} subjects, statistics: {', '.join(names)}")

    p2f = _load_projector_module()
    subjects = {}
    for d in subject_dirs:
        log.info(f"  building projector for {d.name}")
        subjects[d.name] = SubjectMaps(p2f.Projector(d.name), d)

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        log.info(f"  === {name} ===")
        results[name] = run_statistic(name, subjects, args.alpha,
                                      args.min_subjects,
                                      args.fwe_min_subjects)
        for key, arr in results[name].items():
            np.save(args.out / f"{name}__{key}.npy", arr)

    # Intersection-union conjunctions, per test family.
    for conj, (a, b) in CONJUNCTIONS.items():
        if a not in results or b not in results:
            continue
        log.info(f"  === {conj} (intersection-union of {a}, {b}) ===")
        out = {
            "mean": np.minimum(results[a]["mean"], results[b]["mean"]),
            "n_subjects": results[a]["n_subjects"],
        }
        for family in ("p_perm", "p_prevalence", "p_signflip_fwe"):
            out[family] = np.maximum(results[a][family], results[b][family])
        out["p_perm_fdr"] = fdr(out["p_perm"], args.alpha)
        out["p_prevalence_fdr"] = fdr(out["p_prevalence"], args.alpha)
        for key, arr in out.items():
            np.save(args.out / f"{conj}__{key}.npy", arr)
        results[conj] = out

    log.info("")
    log.info(f"{'statistic':28s} {'mean':>9s} {'perm':>8s} {'prev':>8s} "
             f"{'FWE':>8s}")
    for name, res in results.items():
        def n_sig(key):
            p = res.get(key)
            return 0 if p is None else int(np.nansum(p < args.alpha))
        log.info(f"{name:28s} {np.nanmean(res['mean']):+9.4f} "
                 f"{n_sig('p_perm_fdr'):8,d} {n_sig('p_prevalence_fdr'):8,d} "
                 f"{n_sig('p_signflip_fwe'):8,d}")
    log.info(f"\n-> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
