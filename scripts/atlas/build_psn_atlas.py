#!/usr/bin/env python3
"""
The region atlas of the paper: HCP-MMP1 areas grouped into anatomical regions.

    fsaverage/label/{lh,rh}.HCPMMP1.annot  --psn_regions.csv-->  {lh,rh}.psn.annot

One atlas for everything the paper names. Clusters are already labelled with
HCP-MMP1 areas (group_clusters.py); the region summaries used to be
Desikan-Killiany, which has no superior temporal sulcus except its posterior
banks, splits pars orbitalis away from the AAL3 IFGorb of the irony papers
(DK parsorbitalis covers 31% of it, HCP 47l is its main area), and mixes
gyral names with the HCP names the cluster tables use.

`psn_regions.csv` assigns each of the 180 areas to one region. Language and
auditory cortex is grouped by hand (STG, aSTS, pSTS, MTG, angular and
supramarginal gyrus, the three IFG parts up to the inferior frontal sulcus,
precentral); every other area keeps
its Glasser et al. (2016) cortical section, so the atlas covers the whole
cortex and a ranking of regions never compares a group with a single area.
The section of each area is in the table, from the HCP-MMP1 region list of
the neuroimaging-core-docs atlas page (D. Patterson, U. Arizona).

The annotation is written next to HCPMMP1 in every fsaverage label directory
the pipeline reads (the project's own and the FreeSurfer install), so
`--atlas psn` / `--annot psn` work in plot_group_inflated.py and
roi_context_curve.py without a path option.

    python scripts/atlas/build_psn_atlas.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel.freesurfer as fsio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLE = HERE / "psn_regions.csv"
LABEL_DIRS = [Path(config.FREESURFER_DIR) / "fsaverage" / "label",
              Path("/usr/local/freesurfer/7.4.1/subjects/fsaverage/label")]


def regions() -> tuple[dict[str, str], list[str]]:
    rows = list(csv.DictReader(open(TABLE)))
    if len(rows) != 180 or len({r["area"].upper() for r in rows}) != 180:
        raise SystemExit(f"{TABLE} must assign each of the 180 areas once")
    order = list(dict.fromkeys(r["region"] for r in rows))
    # the region list writes 7Pl where the annotation has 7PL
    return {r["area"].upper(): r["region"] for r in rows}, order


def build(src: Path, dst: Path, area2region: dict[str, str],
          order: list[str]) -> dict[str, int]:
    labels, ctab, names = fsio.read_annot(str(src))
    names = [n.decode() for n in names]
    # HCP names are '<H>_<area>_ROI'; index 0 is '???' (medial wall)
    lut = np.full(len(names), -1)
    for i, n in enumerate(names):
        if n != "???":
            lut[i] = order.index(area2region[n[2:].removesuffix("_ROI").upper()])
    out = np.where(labels >= 0, lut[np.maximum(labels, 0)], -1)
    rng = np.random.default_rng(0)
    ct = np.zeros((len(order), 5), int)
    ct[:, :3] = rng.integers(40, 220, (len(order), 3))
    ct[:, 4] = ct[:, 0] + ct[:, 1] * 256 + ct[:, 2] * 65536
    fsio.write_annot(str(dst), out, ct, [r.encode() for r in order],
                     fill_ctab=False)
    return {r: int((out == i).sum()) for i, r in enumerate(order)}


def main() -> int:
    area2region, order = regions()
    done = set()
    for d in LABEL_DIRS:
        if not (d / "lh.HCPMMP1.annot").exists() or d.resolve() in done:
            continue
        done.add(d.resolve())
        for hemi in ("lh", "rh"):
            n = build(d / f"{hemi}.HCPMMP1.annot", d / f"{hemi}.psn.annot",
                      area2region, order)
            print(f"{d / f'{hemi}.psn.annot'}: {len(order)} regions, "
                  f"pSTS {n['pSTS']}, aSTS {n['aSTS']}, IFG_orb {n['IFG_orb']}")
    if not done:
        raise SystemExit("no fsaverage label directory with HCPMMP1 found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
