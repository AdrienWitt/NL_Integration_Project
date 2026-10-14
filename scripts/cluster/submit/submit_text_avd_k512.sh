#!/bin/bash
# Text (Llama-3.1-8B L13, k=512) + AVD (3 d) joint fit, cv + holdout, for the
# accuracy table only (no nulls). Same settings as the other joint rows.
cd ~/NL_Integration_Project
sbatch --parsable --exclude=gpu050 \
  --export=ALL,TEXT=llama8b_k512L13,AUDIO=emotion_avd \
  scripts/cluster/encoding_joint.sbatch
