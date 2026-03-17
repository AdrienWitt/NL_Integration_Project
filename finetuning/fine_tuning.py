import os
import argparse
from sklearn.model_selection import train_test_split

from transformers import AutoFeatureExtractor
import json

from utils.dataset import ProsodyBrainDataset
from utils.training import train_model
from utils.config import STIMULI_DIR, PROSODY_DIR, MAIN_DIR, PROSODY_BRAIN_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train self-supervised speech model for prosody (and optionally brain) feature prediction"
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--audio_dir", type=str, default=STIMULI_DIR, required=False,
                        help=f"Directory containing .wav files (default: {STIMULI_DIR})")
    parser.add_argument("--prosody_brain_dir", type=str, default=PROSODY_BRAIN_DIR, required=False,
                        help="Directory with combined prosody+brain JSON files "
                             f"(default: {PROSODY_BRAIN_DIR})")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save models, metrics, splits, etc.")

    # ── Mode ───────────────────────────────────────────────────────────────
    parser.add_argument("--use_brain_pca", action="store_true",
                        help="Use brain PCA components as labels (long format: all subjects). "
                             "If omitted, prosody features are used as labels (each story once).")

    # ── Filters ────────────────────────────────────────────────────────────
    parser.add_argument("--subject_names", type=str, default=None,
                        help="Comma-separated subject IDs to include, e.g. 'sub-01,sub-02'. "
                             "Only relevant with --use_brain_pca.")

    # ── Split ──────────────────────────────────────────────────────────────
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of stories to use for validation")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for train/val story split")

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument("--model_type", type=str, default="wav2vec2",
                        choices=["wav2vec2", "hubert", "wavlm"],
                        help="Base model family")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Specific pretrained checkpoint (overrides --model_type default)")

    # ── Training hyperparameters ───────────────────────────────────────────
    parser.add_argument("--freeze_layers", type=str, default="12",
                        help="Layers to freeze: integer N (first N) or comma-separated list "
                             "(e.g. '0,1,2') or 'none'")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max number of checkpoints to keep")

    # ── Resume ─────────────────────────────────────────────────────────────
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")

    # ── Multi-task learning ────────────────────────────────────────────────
    parser.add_argument("--brain_weight", type=float, default=0.5,
                        help="Weight for brain PCA loss in multi-task learning. "
                             "loss = prosody_loss + brain_weight * brain_pca_loss. "
                             "Only used with --use_brain_pca (default: 0.5)")

    args = parser.parse_args()

    # ────────────────────────────────────────────────────────────────────────
    # Parse freeze_layers
    # ────────────────────────────────────────────────────────────────────────
    freeze_layers = None
    if args.freeze_layers and args.freeze_layers.lower() != "none":
        try:
            if "," in args.freeze_layers:
                freeze_layers = [int(x.strip()) for x in args.freeze_layers.split(",")]
            else:
                freeze_layers = int(args.freeze_layers)
        except ValueError:
            raise ValueError(
                "--freeze_layers must be an integer, comma-separated list, or 'none'"
            )

    # ────────────────────────────────────────────────────────────────────────
    # Parse optional subject filter
    # ────────────────────────────────────────────────────────────────────────
    subject_names = (
        [s.strip() for s in args.subject_names.split(",")]
        if args.subject_names
        else None
    )

# ── Load pre-generated split ─────────────────────────────────────────────
    split_path = os.path.join(MAIN_DIR, "splits", "stories_split.json")
    with open(split_path) as f:
        split = json.load(f)

    train_stories = split["train"]
    val_stories   = split["val"]

    assert not set(train_stories) & set(val_stories), "ERROR: train/val overlap!"
    print(f"Train stories ({len(train_stories)}): {', '.join(train_stories[:5])} …")
    print(f"Val   stories ({len(val_stories)}):   {val_stories}")
    # ────────────────────────────────────────────────────────────────────────
    # Load processor
    # ────────────────────────────────────────────────────────────────────────
    MODEL_MAP = {
        "wav2vec2": "facebook/wav2vec2-large-960h",
        "hubert":   "facebook/hubert-large-ll60k",
        "wavlm":    "microsoft/wavlm-large",
    }
    processor_model = args.base_model_name or MODEL_MAP.get(args.model_type.lower())
    if processor_model is None:
        raise ValueError(
            f"Unknown model_type '{args.model_type}' and no --base_model_name provided"
        )

    print(f"Loading processor: {processor_model}")
    processor = AutoFeatureExtractor.from_pretrained(processor_model)

    # ────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ────────────────────────────────────────────────────────────────────────
    print("\nResolved paths:")
    print(f"  audio_dir         : {args.audio_dir}")
    print(f"  prosody_brain_dir : {args.prosody_brain_dir}")
    print(f"  mode              : {'brain PCA' if args.use_brain_pca else 'prosody-only'}")
    if subject_names:
        print(f"  subject filter    : {subject_names}")

    try:
        json_files = sorted(
            f for f in os.listdir(args.prosody_brain_dir)
            if f.endswith("_prosody+brain-pca.json") and "_all-stories_" not in f
        )
        print(f"  Found {len(json_files)} per-story JSON files "
              f"(examples): {json_files[:5]}")
    except Exception as e:
        print(f"  Error listing prosody_brain_dir: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # Build datasets
    # Scalers are fitted on the training set and reused for validation to
    # prevent data leakage.
    # ────────────────────────────────────────────────────────────────────────
    common_kwargs = dict(
        audio_dir=args.audio_dir,
        prosody_brain_dir=args.prosody_brain_dir,
        processor=processor,
        use_brain_pca=args.use_brain_pca,
        subject_names=subject_names,
    )

    print("\nBuilding training dataset …")
    train_dataset = ProsodyBrainDataset(
        **common_kwargs,
        story_names=train_stories,
    )

    print("\nBuilding validation dataset (reusing train scalers) …")
    val_dataset = ProsodyBrainDataset(
        **common_kwargs,
        story_names=val_stories,
        scalers=train_dataset.get_fitted_scalers(),   # ← no leakage
    )

    print(f"\nTrain set : {len(train_dataset):,} windows")
    print(f"Val   set : {len(val_dataset):,} windows")
    print(f"Label dim : {train_dataset.label_dim}  "
          f"({'brain PCA components' if args.use_brain_pca else 'prosody features'})")

    # ────────────────────────────────────────────────────────────────────────
    # Launch training
    # ────────────────────────────────────────────────────────────────────────
    resume_path = (
        os.path.join(MAIN_DIR, args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )

    train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        model_type=args.model_type,
        base_model_name=args.base_model_name,
        num_layers_to_freeze=freeze_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=resume_path,
        brain_weight=args.brain_weight,
    )