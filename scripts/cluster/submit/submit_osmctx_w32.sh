#!/bin/bash
# 2026-10-15. eGeMAPS at R=0.5/W=32, so the acoustic context axis has six points for both
# audio bands (as the text band). W=16 took 4.7 h of CPU for the longest story set -> 12 h.
set -e
cd ~/NL_Integration_Project
E=$(sbatch --parsable --array=0 --time=12:00:00 --export=ALL,R=0.5,W=32 --output=logs/osmctx_ext32_%a_out --error=logs/osmctx_ext32_%a_err scripts/cluster/osmctx_extract.sbatch)
S=$(sbatch --parsable --dependency=afterok:$E --export=ALL,VARIANTS=opensmile_r0p5_w32 \
  --output=logs/osmctx_sw32_%a_out --error=logs/osmctx_sw32_%a_err scripts/cluster/osmctx_sweep.sbatch)
echo "extract $E  sweep $S"
