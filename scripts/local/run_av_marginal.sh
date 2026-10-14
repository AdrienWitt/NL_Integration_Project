#!/bin/bash
# A+V marginalfixed nulls (Baobab 13611720, submitted 2026-10-14) -> av__marginalfixed
# group maps + clusters + flatmaps, same chain as avd_marginal. Waits for the job.
set -e
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
R=NL_Integration_Project/results/encoding
N=/root/prosody_nulls; G=results/group; JOB=13611720
while ssh -o BatchMode=yes baobab "squeue -h -j $JOB" 2>/dev/null | grep -q .; do sleep 120; done
ssh -o BatchMode=yes baobab "sacct -X -n -j $JOB -o JobID,State" | tee $N/av_marginal_sacct.txt
REM=$(ssh -o BatchMode=yes baobab "cd $R; ls -d bands_arousal-valence*marginalfixed" | head -1)
echo "### $(date +%T) pull $REM"
mkdir -p $N/av_marginal
rsync -a baobab:$R/$REM/permutation/ $N/av_marginal/
n=$(ls $N/av_marginal/*/report.json | wc -l); echo "  $n report.json"; [ "$n" = 9 ]
ST=$(python3 -c "import json;print(' '.join(json.load(open('$N/av_marginal/UTS01/report.json'))['results']))")
echo "  stats: $ST"
cd "$P"; export PYTHONPATH=.
echo "### $(date +%T) group av__marginalfixed"
python3 scripts/group_maps.py --perm-dir $N/av_marginal --out $G/av__marginalfixed > $N/group_av_marginal.log 2>&1
python3 scripts/group_maps.py --out $G/av__marginalfixed --contrasts >> $N/group_av_marginal.log 2>&1
tail -5 $N/group_av_marginal.log
echo "### $(date +%T) clusters"
python3 scripts/group_clusters.py --perm-dir $N/av_marginal --group-dir $G/av__marginalfixed --stats $ST > $N/clusters_av_marginal.log 2>&1
tail -3 $N/clusters_av_marginal.log
echo "### $(date +%T) flatmaps + inflated"
for s in plot_group_maps plot_group_inflated; do
  python3 scripts/visualisation/$s.py --root $G/av__marginalfixed --test p_tfce_perm_fwe --stats $ST > $N/${s}_av_marginal.log 2>&1 || echo "$s failed, see $N/${s}_av_marginal.log"
done
echo ALLDONE
