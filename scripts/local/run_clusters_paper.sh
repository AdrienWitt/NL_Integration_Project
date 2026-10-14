#!/bin/bash
# Clusters (group-permutation null, z>2.33/3.09 extent+mass, TFCE) for every paper map.
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
P="/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
N=/root/prosody_nulls; G=results/group
until grep -q ALLDONE $N/run_opensmile.log; do sleep 60; done
# AVD conditional ("each") nulls are not local yet
if [ ! -f $N/avd_each/UTS09/report.json ]; then
  mkdir -p $N/avd_each
  rsync -a baobab:NL_Integration_Project/results/encoding/bands_arousal-dominance-valence__shuffle_each__null/permutation/ $N/avd_each/ || echo "rsync avd FAILED"
fi
cd "$P"
cl() { echo "=== $1 $(date -Is)"; shift; PYTHONPATH=. python3 scripts/group_clusters.py "$@" 2>&1 | grep -E "vertices significant|clusters,|^tfce|^z|Error|Traceback|->" ; }
cl osm_conditional --perm-dir $N/llama8b_osm_conditional --group-dir $G/llama8bk1024L13__opensmile__conditional --marginal-dir $G/llama8bk1024L13__opensmile__marginalfixed
cl emo_vs_osm --perm-dir $N/emo_vs_osm --group-dir $G/emotionL11_vs_opensmile__each --stats delta_emotion_given_opensmile delta_opensmile_given_emotion
cl osm_vs_avd --perm-dir $N/osm_vs_avd --group-dir $G/opensmile_vs_avd__each --stats delta_opensmile_given_avd delta_avd_given_opensmile
cl avd_each --perm-dir $N/avd_each --group-dir $G/avd__conditional --stats "delta_valence_given_arousal+dominance" "delta_arousal_given_dominance+valence" "delta_dominance_given_arousal+valence"
cl emo_marginal --perm-dir $N/llama8b_marginal --group-dir $G/llama8bk1024L13__base_emotionL11__marginalfixed --stats text audio
cl osm_marginal --perm-dir $N/llama8b_osm_marginal --group-dir $G/llama8bk1024L13__opensmile__marginalfixed --stats audio
cl avd_marginal --perm-dir $N/avd_marginal --group-dir $G/avd__marginalfixed --stats joint arousal dominance valence
echo ALLDONE
