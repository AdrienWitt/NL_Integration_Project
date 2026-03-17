"""
Visualize encoding scores in fsaverage space using nilearn.
No pycortex required — works on Windows.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORRS_DIR  = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\encoding\results\opensmile_all_stories"
OUT_DIR    = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\encoding\results\opensmile_all_stories\plots"
PERCENTILE = 95

# ---------------------------------------------------------------------------

def load_fsaverage_corrs(corrs_dir):
    subjects = sorted(
        d for d in os.listdir(corrs_dir)
        if os.path.isdir(os.path.join(corrs_dir, d))
        and d.startswith("UTS")
        and os.path.exists(os.path.join(corrs_dir, d, "corrs_fsaverage.npy"))
    )
    print(f"Found {len(subjects)} subjects: {subjects}")

    all_corrs = []
    for subject in subjects:
        path  = os.path.join(corrs_dir, subject, "corrs_fsaverage.npy")
        corrs = np.load(path).ravel()
        all_corrs.append(corrs)
        print(f"  {subject}: min={corrs.min():.3f}, max={corrs.max():.3f}, "
              f"mean={corrs.mean():.3f}, n_vertices={corrs.shape[0]:,}")

    return subjects, np.stack(all_corrs)  # (n_subjects, n_vertices)


def get_n_lh(fsaverage):
    """Get number of left hemisphere vertices from fsaverage pial surface."""
    surf = nib.load(fsaverage.pial_left)
    return surf.darrays[0].data.shape[0]


def plot_brain_surface(data_lh, data_rh, fsaverage, title, out_path,
                       cmap="hot", vmin=None, vmax=None, threshold=None):
    """Plot both hemispheres × both views in a 2×2 grid."""
    vmin = vmin if vmin is not None else 0
    vmax = vmax if vmax is not None else float(
        np.percentile(np.concatenate([data_lh, data_rh]), 99)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             subplot_kw={"projection": "3d"})

    configs = [
        (data_lh, fsaverage.pial_left,  fsaverage.sulc_left,  "left",  "lateral", axes[0, 0]),
        (data_lh, fsaverage.pial_left,  fsaverage.sulc_left,  "left",  "medial",  axes[0, 1]),
        (data_rh, fsaverage.pial_right, fsaverage.sulc_right, "right", "lateral", axes[1, 0]),
        (data_rh, fsaverage.pial_right, fsaverage.sulc_right, "right", "medial",  axes[1, 1]),
    ]

    for data, mesh, sulc, hemi, view, ax in configs:
        plotting.plot_surf_stat_map(
            mesh, data,
            hemi=hemi, view=view,
            cmap=cmap, vmax=vmax, threshold=threshold,
            bg_map=sulc,
            axes=ax, colorbar=False,
        )
        ax.set_title(f"{hemi} — {view}", fontsize=9)

    fig.suptitle(title, fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02,
                 label="Encoding r")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading fsaverage surface ...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    n_lh      = get_n_lh(fsaverage)
    print(f"  LH vertices: {n_lh:,}")

    subjects, all_corrs = load_fsaverage_corrs(CORRS_DIR)
    n_subjects = len(subjects)

    mean_corrs = all_corrs.mean(axis=0)
    std_corrs  = all_corrs.std(axis=0)
    threshold  = float(np.percentile(mean_corrs, PERCENTILE))
    mask       = mean_corrs >= threshold

    print(f"\nCommon mask: {mask.sum():,} / {len(mask):,} vertices "
          f"(top {100-PERCENTILE}%, threshold r={threshold:.4f})")
    print(f"Mean r inside mask  : {mean_corrs[mask].mean():.4f}")
    print(f"Mean r outside mask : {mean_corrs[~mask].mean():.4f}")


    # ── Plot 3: common mask ───────────────────────────────────────────────
    print("Plot 3: common mask ...")
    mask_float = mask.astype(np.float32)
    plot_brain_surface(
        mask_float[:n_lh], mask_float[n_lh:], fsaverage,
        title=f"Common mask — top {100-PERCENTILE}% vertices (r > {threshold:.3f}, n={mask.sum():,})",
        out_path=os.path.join(OUT_DIR, "common_mask.png"),
        cmap="Reds", vmin=0, vmax=1, threshold=0.5,
    )

    # ── Summary stats ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"Total fsaverage vertices : {len(mean_corrs):,}")
    print(f"Common mask size         : {mask.sum():,} ({mask.mean()*100:.1f}%)")
    print(f"Threshold r              : {threshold:.4f}")
    print(f"Mean r inside mask       : {mean_corrs[mask].mean():.4f}")
    print(f"Mean r outside mask      : {mean_corrs[~mask].mean():.4f}")
    print(f"Max mean r               : {mean_corrs.max():.4f}")
    print(f"\nPer-subject stats:")
    print(f"  {'Subject':8s} {'Mean r':>8s} {'Max r':>8s} {'% > thresh':>12s}")
    print(f"  {'-'*42}")
    for i, subject in enumerate(subjects):
        pct = (all_corrs[i] >= threshold).mean() * 100
        print(f"  {subject:8s} {all_corrs[i].mean():>8.4f} "
              f"{all_corrs[i].max():>8.4f} {pct:>11.1f}%")
    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()