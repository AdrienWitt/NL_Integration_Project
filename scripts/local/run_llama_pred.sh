#!/bin/bash
set -e
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
R=NL_Integration_Project/results/encoding
N=/root/prosody_nulls
loc=llama8b_pred; rem=bands_text-audio-pred-wordrate__shuffle_pred__llamanull
out=llama8bk1024L13__base_emotionL11__pred_given_rest
mkdir -p $N/$loc
rsync -a baobab:$R/$rem/permutation/ $N/$loc/
n=$(ls $N/$loc/*/report.json | wc -l); echo "$n report.json"; [ "$n" = 9 ]
cd "$P"
PYTHONPATH=. python3 scripts/group_maps.py --perm-dir $N/$loc --out results/group/$out > $N/group_$loc.log 2>&1
tail -15 $N/group_$loc.log
echo ALLDONE
