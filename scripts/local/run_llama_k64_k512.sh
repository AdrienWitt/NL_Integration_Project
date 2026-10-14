#!/bin/bash
# Llama-3.1-8B L13 x emotion L11 at k=64 and k=512 (submitted 2026-10-13 with
# ~/NL_Integration_Project/submit_llama_k64_k512.sh): pull the nulls, group maps,
# clusters/TFCE, then the context contrasts and grid over the full k series.
# Same chain as run_llama_grid.sh + run_llama_downstream.sh.
set -e
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
R=NL_Integration_Project/results/encoding
N=/root/prosody_nulls
G=results/group
cd "$P"
export PYTHONPATH=.

for k in 64 512; do
  for kind in conditional marginal; do
    if [ $kind = conditional ]; then
      rem=llama8b_k${k}L13__perlayer_base_emotionL11__shuffle_conditional__null; out=llama8bk${k}L13__base_emotionL11__conditional; extra=""
    else
      rem=llama8b_k${k}L13__perlayer_base_emotionL11__marginalfixed; out=llama8bk${k}L13__base_emotionL11__marginalfixed; extra="--contrasts"
    fi
    loc=llama8b_k${k}_$kind
    echo "### $(date +%T) pull $loc"
    mkdir -p $N/$loc
    [ -f $N/$loc/UTS09/report.json ] || rsync -a baobab:$R/$rem/permutation/ $N/$loc/
    n=$(ls $N/$loc/*/report.json | wc -l); echo "  $n report.json"; [ "$n" = 9 ]
    echo "### $(date +%T) group $out"
    # --contrasts is its own mode (reads an existing --out and returns), so it runs second
    if [ -z "$(ls "$G/$out"/*__mean.npy 2>/dev/null)" ]; then
      python3 scripts/group_maps.py --perm-dir $N/$loc --out $G/$out > $N/group_$loc.log 2>&1
    fi
    if [ -n "$extra" ]; then python3 scripts/group_maps.py --out $G/$out $extra >> $N/group_$loc.log 2>&1; fi
    tail -5 $N/group_$loc.log
  done
  echo "### $(date +%T) clusters k$k"
  python3 scripts/group_clusters.py --perm-dir $N/llama8b_k${k}_conditional \
    --group-dir $G/llama8bk${k}L13__base_emotionL11__conditional \
    --marginal-dir $G/llama8bk${k}L13__base_emotionL11__marginalfixed > $N/clusters_llama8b_k$k.log 2>&1
  python3 scripts/visualisation/group_report.py --root $G/llama8bk${k}L13__base_emotionL11__conditional >> $N/clusters_llama8b_k$k.log 2>&1
  tail -2 $N/clusters_llama8b_k$k.log
done

PAT="$G/llama8bk{k}L13__base_emotionL11__conditional"
echo "### $(date +%T) context contrast"
python3 scripts/context_contrast.py --pattern "$PAT" --out $G/context_contrast_llama \
  --pairs 4:16 16:64 64:256 256:512 512:1024 4:64 4:256 4:512 4:1024 16:256 256:1024 > $N/context_contrast_llama.log 2>&1
KS="4 16 64 256 512 1024"
echo "### $(date +%T) integration grid"
python3 scripts/visualisation/integration_grid.py --ks $KS --pattern "$PAT" \
  --contrast-dir $G/context_contrast_llama --out $G/integration_grid_llama > $N/integration_grid_llama.log 2>&1
echo "### $(date +%T) context report"
python3 scripts/visualisation/context_report.py --ks $KS --pattern "$PAT" \
  --contrast-dir $G/context_contrast_llama --out $G/context_clusters_report_llama.html > $N/context_report_llama.log 2>&1
echo "### $(date +%T) ROI curve"
RUNS=""; for k in $KS; do RUNS="$RUNS --run k$k=$G/llama8bk${k}L13__base_emotionL11__conditional"; done
python3 scripts/roi_context_curve.py $RUNS --out $G/roi_context_curve_llama.csv > $N/roi_curve_llama.log 2>&1
echo ALLDONE

# ---- text-alone scores at k64/k512 for Fig 2c / S Table 1 (mean r over each EV mask,
#      same format as paper_table/llama_context_text_scores.csv)
S=$N/llama_k_scores
for k in 64 512; do
  run=llama8b_k${k}L13__perlayer_base_emotionL11
  mkdir -p $S/$run
  rsync -a --include='*/' --include='*_corrs.npy' --include='voxel_mask.npy' --include='meta.json' \
        --exclude='*' baobab:$R/$run/banded/ $S/$run/
done
python3 - "$S" <<'PY'
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
S = Path(sys.argv[1]); out = Path("results/group/paper_table/llama_context_text_scores.csv")
rows = []
for rd in sorted(S.iterdir()):
    for ev in ("cv", "holdout"):
        for sd in sorted((rd / ev).glob("UTS0*")):
            assert (sd / "meta.json").exists(), f"{sd}: run not finished"
            m = np.load(sd / "voxel_mask.npy").astype(bool)
            for model in ("audio", "joint", "text"):
                r = np.load(sd / f"{model}_corrs.npy")
                rows.append(dict(run=rd.name, eval=ev, subject=sd.name, model=model,
                                 n_vox=int(m.sum()), mean_r=round(float(np.nanmean(r[m])), 6)))
new = pd.DataFrame(rows)
assert len(new) == 2 * 2 * 9 * 3, len(new)
old = pd.read_csv(out)
old = old[~old.run.isin(new.run.unique())]
pd.concat([old, new]).to_csv(out, index=False)
print(new.groupby(["run", "eval", "model"]).mean_r.mean().round(4))
PY

echo "### $(date +%T) sweeps (sweep.csv only) for the selection curves"
mkdir -p $N/sweeps
rsync -a -m --include='*/' --include='sweep.csv' --exclude='*' \
  baobab:$R/semantic_sweep baobab:$R/prosody_sweep $N/sweeps/
echo "### $(date +%T) figures and tables"
python3 scripts/visualisation/plot_roi_context.py > $N/roi_plot_llama.log 2>&1
for ev in cv holdout; do
  python3 scripts/visualisation/plot_selection_curves.py --sweeps $N/sweeps \
    --llama-k results/group/paper_table/llama_context_text_scores.csv --eval $ev > $N/selection_$ev.log 2>&1
done
python3 scripts/paper_table.py > $N/paper_table.log 2>&1
python3 scripts/paper_tables.py > $N/paper_tables.log 2>&1
# S5 needs the CONN renders of k64/k512 (MATLAB, by hand); the other figures re-assemble.
python3 scripts/visualisation/assemble_paper_figures.py > $N/assemble.log 2>&1 || echo "assemble: see $N/assemble.log (S5 waits for the CONN renders)"
python3 scripts/visualisation/assemble_paper_figures.py --test z3.09_extent__p_fwe > $N/assemble_z309.log 2>&1 || echo "assemble z3.09: see $N/assemble_z309.log"
echo ALLDONE2
