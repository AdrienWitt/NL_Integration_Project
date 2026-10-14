#!/bin/bash
# TFCE-FWE-thresholded flatmaps + inflated views for every paper map.
export PATH=/usr/local/freesurfer/7.4.1/bin:$PATH
cd "/mnt/c/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/Prosody_Semantics_NL"
G=results/group; T=p_tfce_perm_fwe
f() { d=$1; shift; echo "=== $d $(date -Is)"
  python3 scripts/visualisation/plot_group_maps.py --root $G/$d --test $T --stats "$@" 2>&1 | grep -E "\->|Error|Traceback|warn|no "
  python3 scripts/visualisation/plot_group_inflated.py --root $G/$d --test $T --stats "$@" 2>&1 | grep -E "\->|Error|Traceback|warn|no " ; }
f llama8bk1024L13__opensmile__marginalfixed audio
f llama8bk1024L13__base_emotionL11__marginalfixed text audio
f avd__marginalfixed joint arousal dominance valence
f avd__conditional "delta_valence_given_arousal+dominance" "delta_arousal_given_dominance+valence" "delta_dominance_given_arousal+valence"
f emotionL11_vs_opensmile__each delta_emotion_given_opensmile delta_opensmile_given_emotion
f opensmile_vs_avd__each delta_opensmile_given_avd delta_avd_given_opensmile
f llama8bk1024L13__base_emotionL11__conditional delta_audio_given_text delta_text_given_audio delta
f llama8bk1024L13__opensmile__conditional delta_audio_given_text delta_text_given_audio delta
echo ALLDONE
