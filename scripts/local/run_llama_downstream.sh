#!/bin/bash
set -e
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
cd "/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
export PYTHONPATH=.
N=/root/prosody_nulls
G=results/group
PAT="$G/llama8bk{k}L13__base_emotionL11__conditional"
for k in 4 16 256; do
  echo "### $(date +%T) clusters k$k"
  python3 scripts/group_clusters.py --perm-dir $N/llama8b_k${k}_conditional \
    --group-dir $G/llama8bk${k}L13__base_emotionL11__conditional \
    --marginal-dir $G/llama8bk${k}L13__base_emotionL11__marginalfixed > $N/clusters_llama8b_k$k.log 2>&1
  python3 scripts/visualisation/group_report.py --root $G/llama8bk${k}L13__base_emotionL11__conditional >> $N/clusters_llama8b_k$k.log 2>&1
  tail -2 $N/clusters_llama8b_k$k.log
done
echo "### $(date +%T) context contrast"
python3 scripts/context_contrast.py --pattern "$PAT" --out $G/context_contrast_llama \
  --pairs 4:16 4:256 16:256 256:1024 4:1024 > $N/context_contrast_llama.log 2>&1
echo "### $(date +%T) integration grid"
python3 scripts/visualisation/integration_grid.py --ks 4 16 256 1024 --pattern "$PAT" \
  --contrast-dir $G/context_contrast_llama --out $G/integration_grid_llama > $N/integration_grid_llama.log 2>&1
echo "### $(date +%T) context report"
python3 scripts/visualisation/context_report.py --ks 4 16 256 1024 --pattern "$PAT" \
  --contrast-dir $G/context_contrast_llama --out $G/context_clusters_report_llama.html > $N/context_report_llama.log 2>&1
echo "### $(date +%T) partition"
python3 scripts/visualisation/make_partition_maps.py \
  --conditional $G/llama8bk1024L13__base_emotionL11__conditional \
  --marginal $G/llama8bk1024L13__base_emotionL11__marginalfixed \
  --out $G/llama8bk1024L13__base_emotionL11__partition > $N/partition_llama.log 2>&1
cat $N/partition_llama.log | tail -20
echo ALLDONE
