#!/bin/bash
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
R=NL_Integration_Project/results/encoding
N=/root/prosody_nulls
run() {  # loc rem out
  loc=$1; rem=$2; out=$3
  echo "=== $loc $(date -Is)"
  mkdir -p $N/$loc
  rsync -a baobab:$R/$rem/permutation/ $N/$loc/ || { echo "rsync FAILED $loc"; return; }
  n=$(ls $N/$loc/*/report.json | wc -l); echo "$n report.json"; [ "$n" = 9 ] || { echo "INCOMPLETE $loc"; return; }
  cd "$P"
  PYTHONPATH=. python3 scripts/group_maps.py --perm-dir $N/$loc --out results/group/$out > $N/group_$loc.log 2>&1 || { echo "group_maps FAILED $loc"; tail -5 $N/group_$loc.log; return; }
  tail -12 $N/group_$loc.log
  PYTHONPATH=. python3 scripts/visualisation/plot_group_inflated.py --root results/group/$out > $N/inflated_$loc.log 2>&1 || echo "inflated FAILED $loc"
  PYTHONPATH=. python3 scripts/visualisation/plot_group_maps.py --root results/group/$out > $N/flat_$loc.log 2>&1 || echo "flatmaps FAILED $loc"
}
run llama8b_osm_marginal  perlayer_llama8b_k1024L13__opensmile__marginalfixed            llama8bk1024L13__opensmile__marginalfixed
run llama8b_osm_conditional perlayer_llama8b_k1024L13__opensmile__shuffle_conditional__null llama8bk1024L13__opensmile__conditional
run emo_vs_osm            bands_emotion-opensmile__shuffle_each__null                     emotionL11_vs_opensmile__each
run osm_vs_avd            bands_opensmile-avd__shuffle_each__null                         opensmile_vs_avd__each
echo ALLDONE
