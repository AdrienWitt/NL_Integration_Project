"""
Building the design matrix.

The pipeline is deliberately identical for every model (text-only, audio-only,
joint) so that differences in prediction accuracy are attributable to the
feature spaces and not to preprocessing:

  1. trim  `[TR_PAD + trim : -trim]` TRs off each story
  2. z-score each feature space globally, across the concatenated stories
  3. optionally PCA each feature space (fitted on training stories only)
  4. apply FIR delays *per story*, so no story leaks into its neighbour
  5. concatenate the delayed bands side by side

Step 5 is what makes banded ridge possible: each feature space occupies a
contiguous column block, returned as `bands`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from config import TR_PAD
from common.ridge_utils.npp import zscore
from common.ridge_utils.utils import make_delayed


@dataclass
class Design:
    """A design matrix plus everything needed to cross-validate and band it."""

    X: np.ndarray                       #: (n_TRs, n_total_columns)
    bands: Dict[str, slice]             #: band name -> column block of X
    story_ids: np.ndarray               #: (n_TRs,) integer story index per TR
    stories: List[str]                  #: story name for each index
    fitted_pca: Dict[str, PCA] = field(default_factory=dict)

    @property
    def run_onsets(self) -> np.ndarray:
        """First row index of each story — the run structure for CV."""
        changes = np.flatnonzero(np.diff(self.story_ids)) + 1
        return np.concatenate([[0], changes])

    def band_matrix(self, name: str) -> np.ndarray:
        return self.X[:, self.bands[name]]

    def __repr__(self) -> str:
        bands = ", ".join(
            f"{n}={s.stop - s.start}" for n, s in self.bands.items()
        )
        return (f"Design(X={self.X.shape}, stories={len(self.stories)}, "
                f"bands[{bands}])")


def trim_story(arr: np.ndarray, trim: int) -> np.ndarray:
    """Drop `TR_PAD + trim` TRs from the start and `trim` from the end.

    The asymmetry is not a bug: `load_simulated_trfiles` already pads the
    onset grid by `TR_PAD`, so the extra TRs at the start are padding while
    the ones at the end are real but unmodelled.
    """
    if trim <= 0:
        return arr.copy()
    start = TR_PAD + trim
    if arr.shape[0] <= start + trim:
        raise ValueError(
            f"Story too short to trim: {arr.shape[0]} TRs, "
            f"need more than {start + trim}"
        )
    return arr[start:-trim]


def _stack_stories(features: Dict[str, np.ndarray], stories: Sequence[str],
                   trim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Trim and vstack `stories`; also return the per-TR story index."""
    blocks, lengths = [], []
    for story in stories:
        block = trim_story(np.asarray(features[story], dtype=np.float64), trim)
        blocks.append(block)
        lengths.append(block.shape[0])
    story_ids = np.concatenate(
        [np.full(n, i, dtype=int) for i, n in enumerate(lengths)]
    )
    return np.vstack(blocks), story_ids


def build_design(
    stories: Sequence[str],
    feature_spaces: Dict[str, Dict[str, np.ndarray]],
    trim: int = 5,
    ndelays: int = 4,
    use_pca: bool = False,
    n_comps: float = 0.90,
    fitted_pca: Optional[Dict[str, PCA]] = None,
) -> Design:
    """Assemble a delayed, banded design matrix.

    Parameters
    ----------
    stories : sequence of str
        Stories to include, in the order they should be concatenated.
    feature_spaces : dict
        ``{band_name: {story: (n_TRs, n_dim)}}``. Band order fixes column order.
    trim, ndelays : int
        See module docstring. `ndelays` FIR delays of 1..ndelays TRs.
    use_pca : bool
        Reduce each band before delaying. Dimensionality is per band, so the
        text band is not squeezed by the audio band's rank.
    n_comps : float or int
        Passed to `PCA(n_components=...)`: a float <= 1 is explained variance.
    fitted_pca : dict, optional
        Pre-fitted PCA per band, from a training `Design`. Pass this when
        building the *test* design so the test set never refits the projection.

    Returns
    -------
    Design
    """
    if not feature_spaces:
        raise ValueError("feature_spaces is empty")

    stories = list(stories)
    for band, features in feature_spaces.items():
        missing = [s for s in stories if s not in features]
        if missing:
            raise KeyError(f"Band {band!r} is missing stories: {missing}")

    delays = range(1, ndelays + 1)
    fitted_pca = dict(fitted_pca or {})

    delayed_bands: Dict[str, np.ndarray] = {}
    story_ids: Optional[np.ndarray] = None

    for band, features in feature_spaces.items():
        stacked, ids = _stack_stories(features, stories, trim)

        if story_ids is None:
            story_ids = ids
        elif not np.array_equal(story_ids, ids):
            raise ValueError(
                f"Band {band!r} has a different per-story TR count than the "
                f"previous bands — features are not on the same TR grid."
            )

        # Global z-scoring, across stories, before any projection.
        stacked = zscore(stacked)

        if use_pca:
            if band in fitted_pca:
                pca = fitted_pca[band]
                stacked = pca.transform(stacked)
            else:
                comps = int(n_comps) if n_comps > 1 else n_comps
                pca = PCA(n_components=comps)
                stacked = pca.fit_transform(stacked)
                fitted_pca[band] = pca
            print(f"  PCA[{band}]: {stacked.shape[1]} components "
                  f"({fitted_pca[band].explained_variance_ratio_.sum():.3f} "
                  f"variance explained)")

        # Delays per story: a story must not borrow rows from the next one.
        per_story = [
            make_delayed(stacked[story_ids == i], delays)
            for i in range(len(stories))
        ]
        delayed_bands[band] = np.vstack(per_story)

    # Lay the bands out side by side and record their column blocks.
    bands: Dict[str, slice] = {}
    cursor = 0
    for band, block in delayed_bands.items():
        bands[band] = slice(cursor, cursor + block.shape[1])
        cursor += block.shape[1]

    X = np.hstack([delayed_bands[b] for b in delayed_bands]).astype(np.float32)

    return Design(X=X, bands=bands, story_ids=story_ids,
                  stories=stories, fitted_pca=fitted_pca)


def trim_response(resp: np.ndarray, n_feature_trs: int, trim: int) -> np.ndarray:
    """Put a story's response on the same TR grid as its trimmed features.

    Features are sampled on a grid of ``respdict[story] - TR_PAD`` onsets
    (`load_simulated_trfiles` simulates ``resps - pad`` TRs) and are then cut
    to ``[TR_PAD + trim : -trim]``. A stored response is on one of two grids:

    * the same pad-shortened grid  -> ``offset == 0``
    * the raw acquisition grid     -> ``offset == TR_PAD``, the extra rows
      being the padding TRs at the start

    Both are handled by anchoring the two grids at their common end and
    applying the identical cut. Any other offset means features and responses
    were not generated from the same `respdict`, which is a data problem and
    is raised rather than silently patched over.

    Parameters
    ----------
    resp : (n_TRs, n_voxels) array
    n_feature_trs : int
        Rows in this story's *untrimmed* feature array.
    trim : int
        Same `trim` passed to `build_design`.

    Returns
    -------
    (n_TRs_trimmed, n_voxels) array
    """
    offset = resp.shape[0] - n_feature_trs
    if offset not in (0, TR_PAD):
        raise ValueError(
            f"Cannot align response of {resp.shape[0]} TRs to a feature grid "
            f"of {n_feature_trs} TRs: offset {offset} is neither 0 (response "
            f"already on the padded grid) nor {TR_PAD} (raw acquisition "
            f"grid). Features and responses likely come from different "
            f"respdict.json versions."
        )
    if trim <= 0:
        return resp[offset:].copy()

    start = offset + TR_PAD + trim
    end = resp.shape[0] - trim
    if end <= start:
        raise ValueError(
            f"Response too short to trim: {resp.shape[0]} TRs with "
            f"offset={offset}, trim={trim}"
        )
    return resp[start:end]


def response_offset(resp_trs: int, n_feature_trs: int) -> int:
    """Grid offset between a raw response and its feature array (0 or TR_PAD)."""
    return resp_trs - n_feature_trs


def prepare_responses(Y: np.ndarray, zscore_responses: bool = True) -> np.ndarray:
    """NaN-clean and (optionally) z-score a response matrix, then centre it."""
    Y = np.nan_to_num(np.asarray(Y, dtype=np.float64))
    if zscore_responses:
        Y = zscore(Y)
    return Y - Y.mean(0)
