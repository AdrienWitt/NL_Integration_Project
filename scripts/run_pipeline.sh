#!/usr/bin/env bash
# End-to-end pipeline. Each stage is independent — comment out what you have
# already run. Stages 1-3 are prep, 4-5 fine-tuning, 6-8 encoding and stats.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 0. Check paths =="
python -m config

echo "== 1. Story splits (the held-out story is excluded from fine-tuning) =="
python -m prep.make_story_splits
python -m prep.make_encoding_splits

echo "== 2. Baseline feature bands =="
python -m extract.opensmile --out-name opensmile          # 88 eGeMAPS, prosody baseline
python -m extract.gpt2 --type mean                        # semantic band
python -m extract.emotion_avd --output avd --out-name emotion_avd   # 3-d AVD, no fine-tuning

echo "== 3. Fine-tuning targets (needs fsaverage corrs; see README) =="
python -m prep.make_finetune_targets \
    --corrs-dir results/encoding_legacy/opensmile_all_stories \
    --n-pca 3 --percentile 95

# ---------------------------------------------------------------------------
# 4-5. Fine-tuning arms, all on the SAME train stories.
#
#   A  wav2vec2-robust                       best generic 24-layer arm (SSL only)
#   B  emotion                               emotion-pretrained (12 layers)
#   C  wav2vec2-robust --truncate-layers 12  depth-matched control (optional)
#
# A and B answer "which features predict the brain better". C is only needed to
# attribute a difference to emotion pretraining rather than to depth.
# Note wav2vec2-large-960h is deliberately NOT the default arm: it is ASR
# fine-tuned (vocab_size=32) and its top layers discard prosody.
# `python -m finetune.run_finetune --list-models` prints the registry.
# ---------------------------------------------------------------------------
echo "== 4a. Fine-tune wav2vec2-large-robust (arm A) =="
python -m finetune.run_finetune --model wav2vec2-robust \
    --use-brain-pca --brain-weight 0.5 --num-epochs 15 --batch-size 8

echo "== 4b. Fine-tune the emotion model (arm B) =="
python -m finetune.run_finetune --model emotion \
    --use-brain-pca --brain-weight 0.5 --num-epochs 15 --batch-size 8

echo "== 4c. Depth-matched control (arm C, optional) =="
python -m finetune.run_finetune --model wav2vec2-robust --truncate-layers 12 \
    --use-brain-pca --brain-weight 0.5 \
    --num-epochs 15 --batch-size 8

echo "== 5. Features from each fine-tuned encoder =="
# After fine-tuning on prosody, the UPPER layers are the prosody-tuned ones,
# so read high here — unlike the mid-network defaults for the base models.
python -m extract.wav2vec --model results/finetune/wav2vec2_robust_frozen_6_multitask/final_model \
    --layers 18-23 --out-name ft_robust
python -m extract.wav2vec --model results/finetune/emotion_frozen_3_multitask/final_model \
    --layers 6-11 --out-name ft_emotion
python -m extract.wav2vec --model results/finetune/wav2vec2_robust_trunc12_frozen_3_multitask/final_model \
    --layers 6-11 --out-name ft_robust12

echo "== 6. Encoding — one run per audio band, same text band throughout =="
for AUDIO in opensmile emotion_avd ft_robust ft_emotion ft_robust12; do
    echo "-- audio band: $AUDIO"
    python -m encoding.run_encoding --subjects all \
        --text-features gpt2_mean --audio-features "$AUDIO" \
        --backend both --eval holdout --min-ev 0.1
done

echo "== 7. Significance =="
for AUDIO in opensmile emotion_avd ft_robust ft_emotion ft_robust12; do
    python -m stats.run_permutation --subjects all \
        --text-features gpt2_mean --audio-features "$AUDIO" \
        --n-perms 1000 --blocklen 10 --min-ev 0.1
done

echo "== 8. Contrasts and tables =="
for AUDIO in opensmile emotion_avd ft_robust ft_emotion ft_robust12; do
    python -m stats.analysis \
        --features "gpt2_mean__${AUDIO}" --backend banded --eval holdout
done

echo "== Done =="
