#!/bin/bash
# 2026-10-16. Negative controls (S Table 8) refit on the selected model: Llama-8B L13 at
# 512 words x emotion L11 with 16 s of preceding audio. Same recipes as the k1024 controls
# (llamapredk1024, llamawordrate, llamainterpc20), only the bands change:
#   prediction   surprisal + entropy (Llama, k512) | text + audio + word rate   SHUFFLE=pred
#   word rate    text | audio + word rate (Heschl text signal)                 SHUFFLE=text
#   interaction  20x20 PC products | text + audio                              SHUFFLE=inter
set -e
cd ~/NL_Integration_Project
T=llama8b_k512L13; A=perlayer_emoctx_r0p5_w16:11
C="COMPUTE_CAPABILITY_8_0|COMPUTE_CAPABILITY_8_6|COMPUTE_CAPABILITY_8_9|COMPUTE_CAPABILITY_9_0"
X="--parsable --exclude=gpu050"
ENV="module load GCCcore/10.3.0 Python/3.9.5 && . envs/py39/bin/activate && export PYTHONPATH=. DATA_DIR=\$HOME/NL_Integration_Project/data"

# ---- prediction: surprisal at k512 -> split off word rate -> fit -> null
S=$(sbatch $X --array=0 --time=06:00:00 --mem=96G --gres=gpu:1,VramPerGpu:40G --constraint="$C" \
    --export=ALL,HF_HOME=$HOME/scratch/hf,MODEL=NousResearch/Meta-Llama-3.1-8B,KS=512,PE_VECTOR=0,OUTNAMES=surprisal_llama8b_k512 \
    scripts/cluster/extract_surprisal.sbatch)
SP=$(sbatch $X --dependency=afterok:$S --job-name=split_k512 --partition=shared-cpu --time=00:30:00 --mem=16G \
    --output=logs/split_k512_out --error=logs/split_k512_out \
    --wrap="$ENV && python3 scripts/split_surprisal_bands.py --ks 512 --store-prefix surprisal_llama8b_k --pred-prefix surp_pred_llama8b_k --wordrate-name wordrate_llama8b_k512")
P=$(sbatch $X --dependency=afterok:$SP \
    --export=ALL,K=512,TEXT=$T,AUDIO=$A,PRED=surp_pred_llama8b_k512,WORDRATE=wordrate,TAG=llamapredk512emo16 \
    scripts/cluster/encoding_prediction.sbatch)
PN=$(sbatch $X --dependency=aftercorr:$P \
    --export=ALL,TEXT=$T,AUDIO=$A,SHUFFLE=pred,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_k512emo16null,BANDS="text=$T audio=$A pred=surp_pred_llama8b_k512 wordrate=wordrate",MODELS="joint text+audio+wordrate text+audio",FROM_DIR=results/encoding/bands_text-audio-pred-wordrate__llamapredk512emo16 \
    scripts/cluster/permutation.sbatch)
echo "prediction: surprisal $S split $SP fit $P null $PN"

# ---- word rate
W=$(sbatch $X --export=ALL,TEXT=$T,AUDIO=$A,WORDRATE=wordrate,TAG=llamawordratek512emo16 scripts/cluster/encoding_wordrate.sbatch)
WN=$(sbatch $X --dependency=aftercorr:$W \
    --export=ALL,TEXT=$T,AUDIO=$A,SHUFFLE=text,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_k512emo16wrnull,BANDS="text=$T audio=$A wordrate=wordrate",MODELS="joint audio+wordrate text+audio audio",FROM_DIR=results/encoding/bands_text-audio-wordrate__llamawordratek512emo16 \
    scripts/cluster/permutation.sbatch)
echo "word rate: fit $W null $WN"

# ---- interaction: 20x20 PC products of the two selected bands
I=inter_${T}_x_perlayer_emoctx_r0p5_w16L11_pc20
B=$(sbatch $X --job-name=mk_inter_k512 --partition=shared-cpu --time=03:00:00 --mem=64G --cpus-per-task=8 \
    --output=logs/mk_inter_k512_out --error=logs/mk_inter_k512_out \
    --wrap="$ENV && python3 scripts/make_interaction_band.py --text $T --audio $A --n-components 20")
IF=$(sbatch $X --dependency=afterok:$B --export=ALL,TEXT=$T,AUDIO=$A,INTER=$I,TAG=llamainterk512emo16 scripts/cluster/encoding_interaction.sbatch)
IN=$(sbatch $X --dependency=aftercorr:$IF \
    --export=ALL,TEXT=$T,AUDIO=$A,SHUFFLE=inter,N_PERMS=1000,SAVE_NULL=1,TAGSUF=_k512emo16null,BANDS="text=$T audio=$A inter=$I",MODELS="joint text+audio",FROM_DIR=results/encoding/bands_text-audio-inter__llamainterk512emo16 \
    scripts/cluster/permutation.sbatch)
echo "interaction: build $B fit $IF null $IN"
