#!/usr/bin/env python3
"""
CONN surface renders -> publication figures, driven end to end from Python.

    results/group/<run>/export/conn/*.surf.nii          export_group_conn.py
      -> export/conn/fig_mosaic8/<stat>__<test>.surf.png  (+ .range.txt)
                                                        stage 1, MATLAB/CONN
      -> figures_conn/<stat>__<test>.svg (+ png/pdf)    stage 2, figtools

Modelled on the neural_integration Study_3 pipeline (s5_print_contrasts.m ->
s6_build_figures.py) and reusing its pieces rather than copying them:
`conn_print_maps.m` drives CONN's own printer, `figtools` lays out the
8-view mosaic as panels A-F with the house colorbar and title.

Stage 1 launches MATLAB itself, which on this machine needs one workaround:
`matlab -batch` crashes at startup (0xc0000005) inside the VMware Horizon
printer driver (hznprui.dll) when the Windows default printer is a "(vdi)"
one, because MATLAB queries the default printer before running anything. The
default is switched to the local "Microsoft Print to PDF" for the duration of
the run and restored in a `finally`, whatever happens.

Usage
-----
    python scripts/visualisation/conn_figures.py --root results/group/<run>
    python scripts/visualisation/conn_figures.py --root results/group/<run> \\
        --stats delta delta_audio_given_text --export png,pdf
    python scripts/visualisation/conn_figures.py --root ... --skip-print   # re-layout only

Run export_group_conn.py on the same directory first.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent.parent
STUDY3 = (PROJECT.parent / "writing_cowork" / "projects" / "neural_integration"
          / "behavioral_scripts_and_data" / "Study_3")

log = logging.getLogger("conn_figures")

MATLAB = Path("/mnt/c/Program Files/MATLAB/R2023b/bin/matlab.exe")
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
INKSCAPE_WIN = Path("/mnt/c/Program Files/Inkscape/bin/inkscape.com")
TOOLBOX = Path("/mnt/c/Users/wittmann/OneDrive - Université de Genève/Documents/MATLAB/toolbox")
SAFE_PRINTER = "Microsoft Print to PDF"
LAYOUT = "mosaic8"
#: The size of a CONN mosaic8 print that figtools' masks, letters and colorbar
#: are calibrated to (Study_3 build_brain_figure.py). CONN's print size follows
#: the figure's on-screen size, so a VDI session reconnected at a different
#: display scaling changes it -- at 150 % the default 1200x1000 figure came
#: out 4241x3528, and every mask landed on the brains. normalise_mosaic()
#: brings any print back to this layout before figtools sees it.
MOSAIC_SIZE = (3275, 2514)
#: Where each of the 8 views sits in that calibrated mosaic, (x0, y0, x1, y1),
#: column by column, top to bottom: L lat, L med, anterior | superior,
#: inferior | R lat, R med, posterior. Measured on a 3275x2514 print.
REF_VIEWS = [
    (87, 65, 992, 697), (65, 832, 970, 1464), (219, 1687, 842, 2254),
    (1121, 65, 1744, 871), (1121, 1000, 1744, 1806),
    (1873, 59, 2773, 695), (1902, 826, 2802, 1462), (2024, 1687, 2647, 2254),
]

#: Shared by every signed map (see SEQUENTIAL below): a middle fade, and a
#: ramp that starts at a pale tint rather than white, so small values are
#: coloured instead of washing the surface out.
DIVERGING_GAMMA = 0.85
DIVERGING_FLOOR = 0.0


def _gamma_diverging(gamma: float = DIVERGING_GAMMA, floor: float = DIVERGING_FLOOR,
                     n: int = 24) -> list[str]:
    """Hex anchors dark blue -> (0) -> dark red, where the distance from white
    grows as floor + (1 - floor) * |t| ** gamma. With floor > 0 neither half
    touches pure white: the smallest value either side of zero is already a
    pale tint, so no vertex renders as a white patch on the surface."""
    import numpy as np
    def lerp(anchors, u):
        rgb = np.array([[int(a[i:i + 2], 16) for i in (1, 3, 5)] for a in anchors], float)
        x = np.linspace(0, 1, len(rgb))
        return [np.interp(u, x, rgb[:, c]) for c in range(3)]
    blues = ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    reds = ["#ffffff", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"]
    hexs = lambda c: "#%02x%02x%02x" % tuple(int(round(v)) for v in c)
    s = floor + (1 - floor) * np.linspace(0, 1, n) ** gamma
    # Both halves keep their palest end: ramp() splits the anchors in two and
    # interpolates each half separately, so the list must be even and the
    # split must fall between the two pale ends.
    neg = [hexs(lerp(blues, u)) for u in s[::-1]]
    pos = [hexs(lerp(reds, u)) for u in s]
    return neg + pos


# ---- titles ---------------------------------------------------------------
TITLES = {
    "joint": "Joint model (text + audio)",
    "arousal": "Arousal", "dominance": "Dominance", "valence": "Valence",
}
#: Complete titles, used as-is. The partition maps were thresholded when they
#: were built (and zeroed outside significance), then exported with
#: `--test none`, so their `__unthresholded` suffix describes the export, not
#: the map -- a title derived from it would be wrong.
FULL_TITLES = {
    "unique_text": "Text beyond audio (group permutation, q < .05 FDR)",
    "unique_audio": "Audio beyond text (group permutation, q < .05 FDR)",
    "integration": "Integration: both bands add (group permutation, q < .05 FDR)",
    "valence_unique": "Valence beyond arousal + dominance (group permutation, q < .05 FDR)",
    "preference_valence": "Preference, r valence \u2212 r arousal (all covered vertices)",
    "preference_valence_consistent": "Preference, r valence \u2212 r arousal (\u2265 8 of 9 subjects agree)",
    "preference_valence_winner": "Best-predicting dimension: valence or arousal (all covered vertices)",
    "preference_valence_winner_consistent": "Best-predicting dimension: valence or arousal (\u2265 8 of 9 subjects agree)",
    "preference_consistent": "Preference, r text \u2212 r audio (\u2265 8 of 9 subjects agree)",
    "preference_fwe": "Preference, r text \u2212 r audio (sign-flip FWE, p < .05)",
}
#: One sequential hue per magnitude map, truncated to start at a tinted step:
#: the pale end of a full ColorBrewer ramp matches the surface and would read
#: as "no data" rather than "small effect". Zeros are not painted by CONN.
SEQUENTIAL = {
    "unique_text": ["#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"],
    "unique_audio": ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
    "integration": ["#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f", "#3f007d"],
    # Both signed maps share one fade: a pale tint either side of zero, then
    # a gamma climb into blue (arousal / audio, < 0) or red (valence / text, > 0), so
    # the sign reads even where the value is small while the depth of colour
    # still tracks it -- and the two figures read on the same visual scale.
    "preference_valence_winner": _gamma_diverging(DIVERGING_GAMMA),
    "preference_valence_winner_consistent": _gamma_diverging(DIVERGING_GAMMA),
    "preference": _gamma_diverging(DIVERGING_GAMMA),
    "preference_valence": _gamma_diverging(DIVERGING_GAMMA),
    "preference_valence_consistent": _gamma_diverging(DIVERGING_GAMMA),
    "valence_vs_arousal": _gamma_diverging(DIVERGING_GAMMA),
    "valence_unique": ["#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704"],
}

TEST_LABELS = {
    "p_tfce_perm_fwe": "TFCE, group permutation FWE p < .05",
    "z3.09_extent__p_fwe": "z > 3.09 (≈ p < .001), cluster FWE p < .05",
    "p_perm_fdr": "group permutation, q < .05 FDR",
    "p_prevalence_fdr": "prevalence, q < .05 FDR",
    "p_signflip_fwe": "sign-flip FWE, p < .05",
    "unthresholded": "unthresholded",
}


#: Band names as they appear in titles, for the names used in band-vs-band runs.
BAND_NAMES = {"opensmile": "openSMILE", "avd": "AVD", "emotion": "emotion wav2vec2"}


def bands_of(run: str) -> dict[str, str]:
    """What `text` and `audio` mean in a run, read off its directory name
    (`llama8bk<K>L<L>__<audio>__<test>`, or the older `k<K>L<L>__...`)."""
    out = {"text": "text", "audio": "audio", "text_long": "Text band",
           "audio_long": "Audio band"}
    if (m := re.match(r"(llama8b)?k(\d+)L(\d+)__", run)):
        lm = "Llama-3.1-8B" if m[1] else "GPT-2"
        out["text_long"] = f"Text band ({lm}, layer {m[3]}, {m[2]} words of context)"
        if m[2] != "1024" or not m[1]:
            out["text"] = f"text (k = {m[2]})"
    if "__opensmile__" in run:
        out["audio"] = "openSMILE"
        out["audio_long"] = "Audio band (openSMILE eGeMAPS, 88 features)"
    elif "base_emotionL11" in run:
        out["audio_long"] = "Audio band (wav2vec2 emotion, layer 11)"
    return out


def title_for(stat: str, test: str, run: str = "") -> str:
    if stat in FULL_TITLES:
        return FULL_TITLES[stat]
    b = bands_of(run)
    if stat == "preference" and test == "unthresholded":
        return f"Preference, r {b['text']} \u2212 r {b['audio']} (all covered vertices)"
    if stat == "text":
        base = b["text_long"]
    elif stat == "audio":
        base = b["audio_long"]
    elif stat == "joint" and run.startswith("avd"):
        base = "Affective dimensions (arousal + dominance + valence)"
    elif stat == "preference":
        base = f"Preference (r {b['text']} \u2212 r {b['audio']})"
    elif stat == "delta":
        base = f"Integration: {b['text']} and {b['audio']} each add beyond the other"
    elif stat in TITLES:
        base = TITLES[stat]
    elif (m := re.fullmatch(r"delta_(.+)_given_(.+)", stat)):
        pretty = lambda s: " + ".join(b.get(x, BAND_NAMES.get(x, x)) for x in s.split("+"))
        first = pretty(m[1])
        if not first.startswith("openSMILE"):          # keep its own casing
            first = first[0].upper() + first[1:]
        base = f"{first} beyond {pretty(m[2])}"
    else:
        base = stat.replace("_", " ").capitalize()
    return f"{base} ({TEST_LABELS.get(test, test)})"


#: Signed difference maps, printed on a symmetric diverging scale. Must match
#: SIGNED in print_group_conn.m.
SIGNED = ("preference", "valence_vs_arousal")


def is_signed(stat: str) -> bool:
    return stat.startswith(SIGNED)


def colorbar_label(stat: str, run: str = "") -> str:
    if stat.startswith("preference_valence_winner"):
        return "r of the winner (+ valence, \u2212 arousal)"
    if stat.startswith(("preference_valence", "valence_vs_arousal")):
        return "\u0394r (valence \u2212 arousal)"
    if is_signed(stat):
        return f"\u0394r (text \u2212 {bands_of(run)['audio']})"
    if stat.startswith("delta") or stat in ("unique_text", "unique_audio", "integration", "valence_unique"):
        return "\u0394r"
    return "r"


def _interp(anchors: list[str], n: int):
    import numpy as np
    rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in anchors])
    x = np.linspace(0, 1, len(rgb))
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, x, rgb[:, c]) for c in range(3)], axis=1)


def _halves(stat: str):
    a = SEQUENTIAL[stat]
    if len(a) % 2:
        raise ValueError(f"signed colormap {stat} needs an even number of anchors")
    return a[:len(a) // 2], a[len(a) // 2:]


def ramp(stat: str):
    """RGB in [0, 1] for CONN, from the SEQUENTIAL anchors.

    Magnitude maps: 96 rows, CONN's positive half. Signed maps: 192 rows,
    most negative first, as CONN's own bipolar maps are laid out -- a 96-row
    map on a signed statistic would be taken as the positive half only, and
    CONN would build the negative half itself as flipud(cmap(:,[2 3 1])), a
    channel-permuted mirror (conn_mesh_display.m). The two halves are
    interpolated separately, so the colour breaks at zero instead of blending
    through a neutral that would hide the sign where |value| is small.
    """
    import numpy as np
    if is_signed(stat):
        neg, pos = _halves(stat)
        return np.vstack([_interp(neg, 96), _interp(pos, 96)])
    return _interp(SEQUENTIAL[stat], 96)


def bar_stops(stat: str) -> list[tuple[float, str]]:
    """SVG gradient stops drawing the same colormap; signed maps get a hard
    break at 0.5 (two stops at the same offset)."""
    if is_signed(stat):
        neg, pos = _halves(stat)
        k = len(neg) - 1
        return ([(0.5 * i / k, c) for i, c in enumerate(neg)]
                + [(0.5 + 0.5 * i / k, c) for i, c in enumerate(pos)])
    r = ramp(stat)
    return [(o, "#%02x%02x%02x" % tuple(int(round(c * 255))
             for c in r[min(95, int(round(o * 95)))]))
            for o in [i / 12 for i in range(13)]]


def write_colormaps(root: Path) -> None:
    """<stat>.cmap.txt beside the .surf.nii, read by print_group_conn.m."""
    import numpy as np
    cdir = root / "export" / "conn"
    for stat in SEQUENTIAL:
        if any(cdir.glob(f"{stat}__*.surf.nii")):
            np.savetxt(cdir / f"{stat}.cmap.txt", ramp(stat), fmt="%.6f")


# ---- Windows helpers ------------------------------------------------------
def winpath(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], check=True,
                          capture_output=True, text=True).stdout.strip()


def powershell(cmd: str) -> str:
    r = subprocess.run([str(POWERSHELL), "-NoProfile", "-Command", cmd],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"powershell failed: {r.stderr.strip()}")
    return r.stdout.strip()


def default_printer() -> str:
    return powershell("(Get-CimInstance Win32_Printer | ? Default).Name")


def set_default_printer(name: str) -> None:
    safe = name.replace("'", "''")
    powershell(f"Get-CimInstance Win32_Printer -Filter \"Name='{safe}'\" | "
               "Invoke-CimMethod -MethodName SetDefaultPrinter | Out-Null")
    now = default_printer()
    if now != name:
        raise RuntimeError(f"default printer is {now!r}, wanted {name!r}")


# ---- stage 1: MATLAB/CONN -------------------------------------------------
def run_matlab(run: str, stats: list[str] | None, timeout: int,
               tests: list[str] | None = None, vmax: float | None = None) -> None:
    for p in (MATLAB, POWERSHELL):
        if not p.exists():
            raise SystemExit(f"not found: {p}")
    # The .m file and the log must live on the Windows side; the user's temp
    # directory is reachable from both.
    wtemp = Path("/mnt/c/Users/wittmann/AppData/Local/Temp")
    work = Path(tempfile.mkdtemp(prefix="conn_figures_", dir=wtemp))
    fn = "conn_figures_job"
    stats_m = "{" + ",".join(f"'{s}'" for s in (stats or [])) + "}"
    tests_m = "{" + ",".join(f"'{s}'" for s in (tests or [])) + "}"
    vhi_m = "[]" if vmax is None else repr(float(vmax))
    (work / f"{fn}.m").write_text(
        f"function {fn}()\n"
        f"addpath('{winpath(TOOLBOX / 'spm12')}');\n"
        f"addpath('{winpath(TOOLBOX / 'conn')}');\n"
        f"addpath('{winpath(_HERE)}');\n"
        f"print_group_conn('{run}', 'Layout', '{LAYOUT}', 'Stats', {stats_m}, "
        f"'Tests', {tests_m}, 'Vhi', {vhi_m});\n"
        f"end\n", encoding="utf-8")
    logfile = work / "matlab.log"

    original = default_printer()
    swap = original != SAFE_PRINTER
    if swap:
        log.info("default printer %r -> %r for the run", original, SAFE_PRINTER)
        set_default_printer(SAFE_PRINTER)
    t0 = time.time()
    try:
        wd, wl = winpath(work), winpath(logfile)
        args = ",".join(f"'{a}'" for a in ("-sd", wd, "-logfile", wl, "-batch", fn))
        cmd = (f"$p = Start-Process -FilePath '{winpath(MATLAB)}' -ArgumentList "
               f"{args} -Wait -PassThru; exit $p.ExitCode")
        r = subprocess.run([str(POWERSHELL), "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=timeout)
        code = r.returncode
    finally:
        if swap:
            set_default_printer(original)
            log.info("default printer restored to %r", original)

    text = logfile.read_text(errors="replace") if logfile.exists() else ""
    wrote = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("wrote ")]
    bad = [ln.strip() for ln in text.splitlines()
           if "[skip map]" in ln or "[SUSPECT]" in ln or "[no output]" in ln]
    log.info("MATLAB exit %d after %.0f s, %d map(s) printed", code,
             time.time() - t0, len(wrote))
    for ln in bad:
        log.warning("%s", ln)
    if code != 0 or bad:
        tail = "\n".join(ln for ln in text.splitlines()[-25:]
                         if "Name is nonexistent" not in ln)
        raise SystemExit(f"MATLAB stage failed (exit {code}); log {logfile}\n{tail}")
    shutil.rmtree(work, ignore_errors=True)


# ---- stage 2: figtools ----------------------------------------------------
def read_range(path: Path):
    vals = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] in ("vmin", "vmax"):
            vals[parts[0]] = float(parts[1])
    return vals.get("vmin"), vals.get("vmax")


def bluewhitered_stops(n: int = 13) -> list[tuple[float, str]]:
    """CONN's 'bluewhitered' (conn_mesh_display.m), sampled as SVG stops.

    Same formula, so the publication colorbar is the colormap CONN painted
    the surface with -- figtools' jet stops would mislabel a signed map.
    """
    import numpy as np
    z48, o48 = np.zeros(48), np.ones(48)
    up, dn = np.linspace(0, 1, 48), np.linspace(1, 0, 48)
    cmap = np.array([
        np.r_[z48, up, o48, np.linspace(1, .5, 48)],
        np.r_[np.linspace(0, 1, 96), dn, z48],
        np.r_[np.linspace(.5, 1, 48), o48, dn, z48],
    ]).T
    w = np.abs(np.linspace(-1, 1, 192))[:, None]
    cmap = w * cmap + (1 - w)
    stops = []
    for o in np.linspace(0, 1, n):
        r, g, b = cmap[min(191, int(round(o * 191)))]
        stops.append((float(o), f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"))
    return stops


def build_svg(png: Path, title_text: str, vmin: float, vmax: float, label: str,
              diverging: bool = False, stops=None) -> str:
    sys.path.insert(0, str(STUDY3))
    from figtools import Canvas, image, masks, colorbar, letters, title
    import build_brain_figure as bbf       # the calibrated layout constants
    lines, pt = title.wrap(title_text, bbf.PIC[2], bbf.TITLE_PT)
    top = title.extra_top(lines, pt, band=bbf.PIC[1])
    cv = Canvas(bbf.W, bbf.H, top_extra=top)
    cv += image.embed(png, *bbf.PIC)
    cv += masks.corners()
    if diverging or stops:
        jet_stops = colorbar.JET_STOPS
        # jet() reads the module global; swap it for this one bar only
        colorbar.JET_STOPS = stops or bluewhitered_stops()
        try:
            cv += colorbar.jet(vmin, vmax, label=label,
                               gid="bwr" if diverging else "seq", stroke="#808080")
        finally:
            colorbar.JET_STOPS = jet_stops
    else:
        cv += colorbar.jet(vmin, vmax, label=label)
    cv += letters.panel()
    cv += title.block(lines, pt, band_top=-top, band_bot=bbf.PIC[1], cx=bbf.W / 2)
    return cv.render()


def export(svg: Path, fmts: list[str], dpi: int | None = None) -> None:
    """Windows Inkscape first: it has Agency FB, the WSL one falls back."""
    for f in fmts:
        out = svg.with_suffix("." + f)
        extra = [f"--export-dpi={dpi}"] if dpi and f == "png" else []
        if INKSCAPE_WIN.exists():
            cmd = [str(INKSCAPE_WIN), winpath(svg), f"--export-type={f}",
                   *extra, "-o", winpath(out)]
        else:
            cmd = ["inkscape", str(svg), f"--export-type={f}", *extra, "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            log.warning("export %s failed: %s", out.name, r.stderr.strip()[-300:])
        else:
            log.info("  exported %s", out.name)


def _runs(v, min_gap: int) -> list[list[int]]:
    """[start, end) of True runs in a 1-D mask, merging gaps < min_gap."""
    import numpy as np
    idx = np.flatnonzero(v)
    if not idx.size:
        return []
    out = [[int(idx[0]), int(idx[0]) + 1]]
    for i in idx[1:]:
        if i - out[-1][1] < min_gap:
            out[-1][1] = int(i) + 1
        else:
            out.append([int(i), int(i) + 1])
    return out


def normalise_mosaic(png: Path) -> Path:
    """Bring a CONN mosaic8 print of any size back onto the calibrated layout.

    CONN's print size follows the figure's on-screen size, so it changes with
    the VDI display scaling (1200x1000 at 150 % gives 4241x3528). What changes
    is only the size of the views and the gaps between them: every view is
    enlarged by the same factor. So each of the 8 views is found (white gaps
    between columns, then between views within a column), scaled by ONE common
    factor back to the reference size, and placed at its reference position.
    No view is distorted. The L/R labels travel with their nearest view; CONN's
    own colorbar is dropped, as figtools masks it anyway.

    Returns `png` untouched when it already has the calibrated size, else a
    normalised copy under fig_mosaic8/normalised/.
    """
    import numpy as np
    from PIL import Image
    with Image.open(png) as im:
        if im.size == MOSAIC_SIZE:
            return png
        rgb = im.convert("RGB")
    a = np.asarray(rgb.convert("L")) < 245
    W = a.shape[1]
    cols = _runs(a.any(0), int(0.02 * W))
    if len(cols) < 3:
        raise ValueError(f"expected 3 view columns, found {len(cols)}")
    views, labels = [], []
    for c, (x0, x1) in enumerate(cols[:3]):          # a 4th column is the colorbar
        for y0, y1 in _runs(a[:, x0:x1].any(1), 15):
            sub = a[y0:y1, x0:x1]
            ys, xs = np.nonzero(sub)
            box = (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)
            (views if sub.sum() >= 2000 else labels).append((c, box))
    if len(views) != len(REF_VIEWS):
        raise ValueError(f"found {len(views)} views, expected {len(REF_VIEWS)}")
    ratios = [((r[2] - r[0]) / (b[2] - b[0]), (r[3] - r[1]) / (b[3] - b[1]))
              for (_, b), r in zip(views, REF_VIEWS)]
    scale = float(np.median(ratios))
    if max(abs(x / scale - 1) for pair in ratios for x in pair) > 0.01:
        raise ValueError("views are not scaled uniformly (ratios "
                         f"{np.round(ratios, 3).tolist()}); not a mosaic8 print?")
    out = Image.new("RGB", MOSAIC_SIZE, "white")

    def paste(box, ref_origin, view_origin):
        crop = rgb.crop(box)
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        x = ref_origin[0] + round((box[0] - view_origin[0]) * scale)
        y = ref_origin[1] + round((box[1] - view_origin[1]) * scale)
        out.paste(crop.resize(size, Image.LANCZOS), (x, y))

    for (_, box), ref in zip(views, REF_VIEWS):
        paste(box, ref, box)
    for c, box in labels:                             # L / R: follow the nearest view
        cy = (box[1] + box[3]) / 2
        k = min((i for i, (vc, _) in enumerate(views) if vc == c),
                key=lambda i: abs((views[i][1][1] + views[i][1][3]) / 2 - cy))
        paste(box, REF_VIEWS[k], views[k][1])
    dst = png.parent / "normalised" / png.name
    dst.parent.mkdir(exist_ok=True)
    out.save(dst)
    log.info("normalised %s: %dx%d -> %dx%d (views x%.3f)", png.name,
             *rgb.size, *MOSAIC_SIZE, scale)
    return dst


def build_figures(root: Path, stats: list[str] | None, fmts: list[str],
                  dpi: int | None = None, tests: list[str] | None = None,
                  vmax: float | None = None) -> int:
    vmax_shared = vmax
    mosaic_dir = root / "export" / "conn" / f"fig_{LAYOUT}"
    out_dir = root / "figures_conn"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    from PIL import Image
    for png in sorted(mosaic_dir.glob("*.surf.png")):
        stem = png.name[:-len(".surf.png")]
        if stem.startswith("_check"):
            continue                       # the check is looked at, not published
        stat, _, test = stem.partition("__")
        if stats and stat not in stats:
            continue
        if tests and test not in tests:
            continue
        try:
            png = normalise_mosaic(png)
        except ValueError as e:
            log.error("[skip] %s: %s", stem, e)
            continue
        rng = png.with_name(stem + ".surf.range.txt")
        vmin, vmax = read_range(rng) if rng.exists() else (None, None)
        if vmin is None or vmax is None:
            log.warning("[skip] %s: no usable %s", stem, rng.name)
            continue
        # print_group_conn rescales non-signed maps to [0, max], but the range
        # sidecar reads state.Vrange, which still holds the data minimum --
        # a few significant vertices carry a slightly negative group mean
        # (185 in delta_audio_given_text). Label the bar as CONN drew it.
        # preference is printed symmetric about zero (print_group_conn.m).
        diverging = is_signed(stat)
        if diverging:
            m = vmax_shared if vmax_shared is not None else max(abs(vmin), abs(vmax))
            vmin, vmax = -m, m
        else:
            vmin = 0.0
            if vmax_shared is not None:
                vmax = vmax_shared          # printed with Vrange [0 vmax]
        svg = out_dir / f"{stem}.svg"
        stops = bar_stops(stat) if stat in SEQUENTIAL else None
        svg.write_text(build_svg(png, title_for(stat, test, root.name), vmin, vmax,
                                 colorbar_label(stat, root.name), diverging, stops),
                       encoding="utf-8")
        log.info("wrote %s  (%.3g .. %.3g)", svg.relative_to(PROJECT), vmin, vmax)
        export(svg, fmts, dpi)
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="results/group/<run>, with export/conn already written")
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--tests", nargs="*", default=None,
                    help="only these test suffixes, e.g. p_tfce_perm_fwe")
    ap.add_argument("--vmax", type=float, default=None,
                    help="fixed colorbar [0, vmax] for magnitude maps ([-vmax, vmax] for "
                         "signed ones), so panels "
                         "compared with each other share one scale")
    ap.add_argument("--export", default="png", help="comma list: png,pdf or ''")
    ap.add_argument("--skip-print", action="store_true",
                    help="reuse the existing CONN mosaics, redo the layout only")
    ap.add_argument("--dpi", type=int, default=None,
                    help="PNG export resolution (Inkscape --export-dpi)")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = args.root.resolve()
    if not (root / "export" / "conn").is_dir():
        raise SystemExit(f"{root}/export/conn missing: run export_group_conn.py first")
    if not (STUDY3 / "figtools").is_dir():
        raise SystemExit(f"figtools not found under {STUDY3}")
    write_colormaps(root)
    if not args.skip_print:
        run_matlab(root.name, args.stats, args.timeout, args.tests, args.vmax)
    fmts = [f.strip() for f in args.export.split(",") if f.strip()]
    n = build_figures(root, args.stats, fmts, args.dpi, args.tests, args.vmax)
    log.info("%d figure(s) in %s", n, (root / "figures_conn").relative_to(PROJECT))
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
