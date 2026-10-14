#!/bin/bash
# 2026-10-14. Emotion band: acoustic context x readout resolution (extract -> layer sweep).
set -e
cd ~/NL_Integration_Project
E=$(sbatch --parsable scripts/cluster/emoctx_extract.sbatch)
S=$(sbatch --parsable --dependency=afterok:$E scripts/cluster/emoctx_sweep.sbatch)
echo "extract $E  sweep $S"
