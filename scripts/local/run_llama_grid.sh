#!/bin/bash
set -e
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
R=NL_Integration_Project/results/encoding
N=/root/prosody_nulls
# local null dir | remote run dir | group out | extra args
JOBS=(
"llama8b_k4_conditional|llama8b_k4L13__perlayer_base_emotionL11__shuffle_conditional__null|llama8bk4L13__base_emotionL11__conditional|"
"llama8b_k4_marginal|llama8b_k4L13__perlayer_base_emotionL11__marginalfixed|llama8bk4L13__base_emotionL11__marginalfixed|--contrasts"
"llama8b_k16_conditional|llama8b_k16L13__perlayer_base_emotionL11__shuffle_conditional__null|llama8bk16L13__base_emotionL11__conditional|"
"llama8b_k16_marginal|llama8b_k16L13__perlayer_base_emotionL11__marginalfixed|llama8bk16L13__base_emotionL11__marginalfixed|--contrasts"
"llama8b_k256_conditional|perlayer_llama8b_k256L13__perlayer_base_emotionL11__shuffle_conditional__null|llama8bk256L13__base_emotionL11__conditional|"
"llama8b_k256_marginal|perlayer_llama8b_k256L13__perlayer_base_emotionL11__marginalfixed|llama8bk256L13__base_emotionL11__marginalfixed|--contrasts"
"llama8b_inter|bands_text-audio-inter__shuffle_inter__llamanull|llama8bk1024L13__base_emotionL11__inter_pc20|"
)
for j in "${JOBS[@]}"; do
  IFS='|' read -r loc rem out extra <<< "$j"
  echo "### $(date +%T) pull $loc"
  mkdir -p $N/$loc
  rsync -a baobab:$R/$rem/permutation/ $N/$loc/
  n=$(ls $N/$loc/*/report.json | wc -l); echo "  $n report.json"; [ "$n" = 9 ]
  echo "### $(date +%T) group $out"
  cd "$P"
  PYTHONPATH=. python3 scripts/group_maps.py --perm-dir $N/$loc --out results/group/$out $extra > $N/group_$loc.log 2>&1
  tail -15 $N/group_$loc.log
done
echo ALLDONE
