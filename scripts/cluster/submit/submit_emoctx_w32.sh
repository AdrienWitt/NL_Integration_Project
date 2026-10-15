#!/bin/bash
# 2026-10-15. Emotion band: does context keep helping past 16 s? One more point, R=0.5/W=32,
# extraction -> layer sweep (cv + holdout), same settings as submit_emoctx.sh.
set -e
cd ~/NL_Integration_Project
E=$(sbatch --parsable --array=0 --export=ALL,R=0.5,W=32 --output=logs/emoctx_ext32_%a_out --error=logs/emoctx_ext32_%a_err scripts/cluster/emoctx_extract.sbatch)
S=$(sbatch --parsable --array=0-8 --dependency=afterok:$E --export=ALL,STORE_LIST=perlayer_emoctx_r0p5_w32 \
  --output=logs/emoctx_sw32_%a_out --error=logs/emoctx_sw32_%a_err scripts/cluster/emoctx_sweep.sbatch)
echo "extract $E  sweep $S"
