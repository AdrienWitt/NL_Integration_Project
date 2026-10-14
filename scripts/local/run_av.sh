#!/bin/bash
# Arousal + valence two-band model (dominance dropped, r=.963 with arousal):
# fit 13578199 + per-band Draper-Stoneman nulls 13578200 (submitted 2026-10-13,
# ~/NL_Integration_Project/submit_av_llamalayers.sh). Same chain as avd_each.
set -e
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
R=NL_Integration_Project/results/encoding
N=/root/prosody_nulls; G=results/group
ST="delta_valence_given_arousal delta_arousal_given_valence"
echo "### $(date +%T) pull av_each"
mkdir -p $N/av_each
rsync -a baobab:$R/bands_arousal-valence__shuffle_each__null/permutation/ $N/av_each/
n=$(ls $N/av_each/*/report.json | wc -l); echo "  $n report.json"; [ "$n" = 9 ]
cd "$P"; export PYTHONPATH=.
echo "### $(date +%T) group av__conditional"
python3 scripts/group_maps.py --perm-dir $N/av_each --out $G/av__conditional > $N/group_av_each.log 2>&1
tail -5 $N/group_av_each.log
echo "### $(date +%T) clusters"
python3 scripts/group_clusters.py --perm-dir $N/av_each --group-dir $G/av__conditional --stats $ST > $N/clusters_av_each.log 2>&1
tail -3 $N/clusters_av_each.log
echo "### $(date +%T) flatmaps + inflated (TFCE FWE)"
for s in plot_group_maps plot_group_inflated; do
  python3 scripts/visualisation/$s.py --root $G/av__conditional --test p_tfce_perm_fwe --stats $ST > $N/${s}_av.log 2>&1 || echo "$s failed, see $N/${s}_av.log"
done
echo ALLDONE
