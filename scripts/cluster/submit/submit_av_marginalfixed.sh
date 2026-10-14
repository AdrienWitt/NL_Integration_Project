#!/bin/bash
# 2026-10-14. Marginal (shuffle both, fixed published alphas) nulls for the two-band
# arousal + valence fit (13578199), so Fig 3 can show the A+V joint map instead of
# the 3-d AVD one. Same recipe as bands_arousal-dominance-valence__marginalfixed.
set -e
cd ~/NL_Integration_Project
AV="arousal=emotion_arousal valence=emotion_valence"
M=$(sbatch --parsable --exclude=gpu050 \
  --output=logs/perm_avmf_%a_out --error=logs/perm_avmf_%a_err \
  --export=ALL,BANDS="$AV",SHUFFLE=both,FROM_DIR=results/encoding/bands_arousal-valence,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_marginalfixed \
  scripts/cluster/permutation.sbatch)
echo "arousal+valence marginalfixed $M"
