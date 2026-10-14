#!/bin/bash
# 2026-10-14. Main text context moves from k=1024 to k=512 (selection rule: best cv
# mean and the >=8/9 rule both pick k512). The emotion x Llama joint at k512 exists
# (13572213-15); this is the eGeMAPS x Llama counterpart (S Fig integration eGeMAPS):
# joint (cv+holdout) -> conditional nulls + marginalfixed nulls, same settings as k1024.
set -e
cd ~/NL_Integration_Project
J=$(sbatch --parsable --exclude=gpu050 --export=ALL,TEXT=llama8b_k512L13,AUDIO=opensmile scripts/cluster/encoding_joint.sbatch)
C=$(sbatch --parsable --exclude=gpu050 --dependency=aftercorr:$J --export=ALL,TEXT=llama8b_k512L13,AUDIO=opensmile,N_PERMS=1000,SHUFFLE=conditional,SAVE_NULL=1,TAGSUF=_null scripts/cluster/permutation.sbatch)
M=$(sbatch --parsable --exclude=gpu050 --dependency=aftercorr:$J --export=ALL,TEXT=llama8b_k512L13,AUDIO=opensmile,N_PERMS=1000,SHUFFLE=both,SAVE_NULL=1,TAGSUF=_marginalfixed scripts/cluster/permutation.sbatch)
echo "osm k512 joint $J conditional $C marginalfixed $M"
