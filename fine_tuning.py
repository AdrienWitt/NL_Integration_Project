import os
import argparse
from sklearn.model_selection import train_test_split

from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

from model_prosody import ProsodyDataset, train_model   # your module with the generalized classes
from encoding.config import DATA_DIR   # if still needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train self-supervised speech model for prosody feature prediction")
    
    # Required paths
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing .wav files")
    parser.add_argument("--prosody_dir", type=str, required=True,
                        help="Directory containing _opensmile_tr_aligned.json files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save models, metrics, splits, etc.")

    # Dataset / split options
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of stories to use for validation")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for train/val story split")

    # Model selection
    parser.add_argument("--model_type", type=str, default="wav2vec2",
                        choices=["wav2vec2", "hubert", "wavlm"],
                        help="Base model family (wav2vec2, hubert, wavlm)")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Specific pretrained checkpoint (overrides model_type default)")

    # Training hyperparameters
    parser.add_argument("--freeze_layers", type=str, default="8",
                        help="Layers to freeze: number N (first N layers) or comma-separated list (e.g. '0,1,2')")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate (3e-5 often good for large models)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Max number of checkpoints to keep")

    # Resume & advanced options
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--use_pca", action="store_true",
                        help="Apply PCA dimensionality reduction to target features")
    parser.add_argument("--pca_threshold", type=float, default=0.90,
                        help="Explained variance ratio threshold for PCA")

    args = parser.parse_args()

    # ────────────────────────────────────────────────────────────────
    # Parse freeze_layers
    # ────────────────────────────────────────────────────────────────
    freeze_layers = None
    if args.freeze_layers is not None and args.freeze_layers.lower() != "none":
        try:
            if ',' in args.freeze_layers:
                freeze_layers = [int(x.strip()) for x in args.freeze_layers.split(',')]
            else:
                freeze_layers = int(args.freeze_layers)
        except ValueError:
            raise ValueError(
                "freeze_layers must be: integer (e.g. 8), comma-separated list (e.g. 0,1,2), or 'none'"
            )

    # ────────────────────────────────────────────────────────────────
    # Get all story names from audio files
    # ────────────────────────────────────────────────────────────────
    audio_files = sorted(f for f in os.listdir(args.audio_dir) if f.lower().endswith(".wav"))
    story_names = [os.path.splitext(f)[0] for f in audio_files]
    story_names = story_names[0:3]

    if not story_names:
        raise FileNotFoundError(f"No .wav files found in {args.audio_dir}")

    # ────────────────────────────────────────────────────────────────
    # Train / val split by story (speaker/story independent)
    # ────────────────────────────────────────────────────────────────
    train_stories, val_stories = train_test_split(
        story_names,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(f"Training stories ({len(train_stories)}): {', '.join(train_stories[:5])} …")
    print(f"Validation stories ({len(val_stories)}): {', '.join(val_stories[:5])} …")

    # Save split for reproducibility
    split_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "train_stories.txt"), "w") as f:
        f.write("\n".join(train_stories) + "\n")
    with open(os.path.join(split_dir, "val_stories.txt"), "w") as f:
        f.write("\n".join(val_stories) + "\n")

    # ────────────────────────────────────────────────────────────────
    # Load appropriate processor
    # ────────────────────────────────────────────────────────────────
    if args.base_model_name:
        processor_model = args.base_model_name
    else:
        MODEL_MAP = {
            "wav2vec2": "facebook/wav2vec2-large-960h",
            "hubert":   "facebook/hubert-large-ll60k",
            "wavlm":    "microsoft/wavlm-large",
        }
        processor_model = MODEL_MAP.get(args.model_type.lower())
        if processor_model is None:
            raise ValueError(f"Unknown model_type {args.model_type} and no --base_model_name provided")

    print(f"Loading processor for: {processor_model}")
    processor = AutoFeatureExtractor.from_pretrained(processor_model)

    # ────────────────────────────────────────────────────────────────
    # Create datasets
    # ────────────────────────────────────────────────────────────────
    common_kwargs = {
        "audio_dir": args.audio_dir,
        "prosody_dir": args.prosody_dir,
        "processor": processor,
        "use_pca": args.use_pca,
        "pca_threshold": args.pca_threshold,
    }
    
    train_dataset = ProsodyDataset(
        **common_kwargs,
        story_names=train_stories,
    )
    val_dataset = ProsodyDataset(
        **common_kwargs,
        story_names=val_stories,
    )

    print(f"Train set size: {len(train_dataset)} windows")
    print(f"Val   set size: {len(val_dataset)} windows")

    # ────────────────────────────────────────────────────────────────
    # Launch training
    # ────────────────────────────────────────────────────────────────
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
        resume_from_checkpoint=args.resume_from_checkpoint,
    )