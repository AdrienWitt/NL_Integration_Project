#!/bin/bash
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
cd "/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
N=/root/prosody_nulls; G=results/group; T=p_tfce_perm_fwe
# 1. null calibration, final models (outputs next to the group maps)
for r in "llama8b_conditional $G/llama8bk1024L13__base_emotionL11__conditional" \
         "llama8b_osm_conditional $G/llama8bk1024L13__opensmile__conditional"; do
  set -- $r; echo "=== diag $1 $(date -Is)"
  PYTHONPATH=. python3 -m stats.diagnose_permutation --perm-dir $N/$1 --out $2/diagnostics 2>&1 | grep -vE "^\S+ \S+ \| INFO" | tail -20
done
# 2. CONN surfaces, TFCE-FWE thresholded, every paper map
e() { d=$1; shift; echo "=== conn $d"; python3 scripts/visualisation/export_group_conn.py --root $G/$d --test $T --stats "$@" 2>&1 | grep -E "wrote|warn|Error|Traceback"; }
e llama8bk1024L13__opensmile__marginalfixed audio
e llama8bk1024L13__base_emotionL11__marginalfixed text audio
e avd__marginalfixed joint arousal dominance valence
e avd__conditional "delta_valence_given_arousal+dominance" "delta_arousal_given_dominance+valence" "delta_dominance_given_arousal+valence"
e emotionL11_vs_opensmile__each delta_emotion_given_opensmile delta_opensmile_given_emotion
e opensmile_vs_avd__each delta_opensmile_given_avd delta_avd_given_opensmile
e llama8bk1024L13__base_emotionL11__conditional delta_audio_given_text delta_text_given_audio delta
e llama8bk1024L13__opensmile__conditional delta_audio_given_text delta_text_given_audio delta
for k in 4 16 256; do e llama8bk${k}L13__base_emotionL11__conditional delta_audio_given_text delta_text_given_audio delta; done
echo "=== conn preference"; python3 scripts/visualisation/export_group_conn.py --root $G/llama8bk1024L13__base_emotionL11__marginalfixed --stats preference --test none 2>&1 | grep -E "wrote|Error"
echo ALLDONE
