#!/bin/bash
# 2026-10-14: main text band moves from k=1024 to k=512 (selection rule).
# Waits for (1) run_llama_k64_k512.sh (builds the emotion x Llama k512 group maps)
# and (2) Baobab 13612672-74 (eGeMAPS x Llama k512 joint + nulls), then builds
# everything the main/SI maps need at k512. Code/text switch (EMO/OSM constants,
# figures, tables, manuscript) is done afterwards, by hand.
set -e
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
R=NL_Integration_Project/results/encoding
N=/root/prosody_nulls; G=results/group; T=p_tfce_perm_fwe
EMO=llama8bk512L13__base_emotionL11; OSM=llama8bk512L13__opensmile
until grep -q ALLDONE2 $N/run_llama_k64_k512.log; do sleep 120; done
while ssh -o BatchMode=yes baobab "squeue -h -j 13612672,13612673,13612674" 2>/dev/null | grep -q .; do sleep 120; done
ssh -o BatchMode=yes baobab "sacct -X -n -j 13612672,13612673,13612674 -o JobID,State" | tee $N/osm_k512_sacct.txt
cd "$P"; export PYTHONPATH=.
for kind in conditional marginal; do
  if [ $kind = conditional ]; then rem=llama8b_k512L13__opensmile__shuffle_conditional__null; out=$OSM"__conditional"
  else rem=llama8b_k512L13__opensmile__marginalfixed; out=$OSM"__marginalfixed"; fi
  loc=osm_k512_$kind
  echo "### $(date +%T) pull $loc"
  mkdir -p $N/$loc; rsync -a baobab:$R/$rem/permutation/ $N/$loc/
  n=$(ls $N/$loc/*/report.json | wc -l); echo "  $n report.json"; [ "$n" = 9 ]
  echo "### $(date +%T) group $out"
  python3 scripts/group_maps.py --perm-dir $N/$loc --out $G/$out > $N/group_$loc.log 2>&1
  [ $kind = marginal ] && python3 scripts/group_maps.py --out $G/$out --contrasts >> $N/group_$loc.log 2>&1
  tail -5 $N/group_$loc.log
done
echo "### $(date +%T) clusters eGeMAPS k512"
python3 scripts/group_clusters.py --perm-dir $N/osm_k512_conditional --group-dir $G/$OSM"__conditional" \
  --marginal-dir $G/$OSM"__marginalfixed" > $N/clusters_osm_k512.log 2>&1
python3 scripts/group_clusters.py --perm-dir $N/osm_k512_marginal --group-dir $G/$OSM"__marginalfixed" --stats audio text >> $N/clusters_osm_k512.log 2>&1
python3 scripts/group_clusters.py --perm-dir $N/llama8b_k512_marginal --group-dir $G/$EMO"__marginalfixed" --stats text audio > $N/clusters_emo_k512_marginal.log 2>&1
tail -3 $N/clusters_osm_k512.log $N/clusters_emo_k512_marginal.log
echo "### $(date +%T) null calibration (S Table 7)"
python3 -m stats.diagnose_permutation --perm-dir $N/llama8b_k512_conditional --out $G/$EMO"__conditional"/diagnostics > $N/diag_emo_k512.log 2>&1
python3 -m stats.diagnose_permutation --perm-dir $N/osm_k512_conditional --out $G/$OSM"__conditional"/diagnostics > $N/diag_osm_k512.log 2>&1
echo "### $(date +%T) partition (preference map, Fig 7d / Data 2-3)"
python3 scripts/visualisation/make_partition_maps.py --conditional $G/$EMO"__conditional" \
  --marginal $G/$EMO"__marginalfixed" --out $G/$EMO"__partition" > $N/partition_k512.log 2>&1
tail -3 $N/partition_k512.log
echo "### $(date +%T) within-subject scores (eGeMAPS x Llama k512)"
S=$N/llama_k_scores; run=llama8b_k512L13__opensmile; mkdir -p $S/$run
rsync -a --include='*/' --include='*_corrs.npy' --include='voxel_mask.npy' --include='meta.json' --exclude='*' baobab:$R/$run/banded/ $S/$run/
echo "### $(date +%T) flatmaps + inflated"
f() { d=$1; shift
  python3 scripts/visualisation/plot_group_maps.py --root $G/$d --test $T --stats "$@" > $N/flat_$d.log 2>&1 || echo "flat $d failed"
  python3 scripts/visualisation/plot_group_inflated.py --root $G/$d --test $T --stats "$@" > $N/infl_$d.log 2>&1 || echo "inflated $d failed"; }
f $EMO"__marginalfixed" text audio
f $EMO"__conditional" delta_audio_given_text delta_text_given_audio delta
f $OSM"__marginalfixed" audio
f $OSM"__conditional" delta_audio_given_text delta_text_given_audio delta
echo ALLDONE
