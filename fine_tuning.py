import os
from wav2vec_prosody import ProsodyDataset, train_model
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from encoding.config import DATA_DIR


audio_dir = os.path.join(DATA_DIR, "stimuli_16k")
prosody_dir = os.path.join(DATA_DIR,"features/prosody/opensmile")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
story_names = [f.replace(".wav", "") for f in os.listdir(audio_dir)]
output_dir = os.path.join(DATA_DIR, "model_output")

# from argparse import Namespace

# args = Namespace(**{
#     "audio_dir": audio_dir,
#     "prosody_dir": prosody_dir,
#     "output_dir": "C:/Users/adywi/OneDrive - unige.ch/Documents/Sarcasm_experiment/NL_Project/model_output",
#     "test_size": 0.2,
#     "random_state": 42,
#     "freeze_layers": None,  # Change to "0,1,2" to freeze specific layers
#     "learning_rate": 1e-4,
#     "batch_size": 8,
#     "num_epochs": 10,
#     "patience": 3,
#     "save_total_limit": 3,
# })

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train wav2vec model for prosody prediction")
    parser.add_argument("--audio_dir", type=str, required=True, 
                      help="Directory containing .wav files")
    parser.add_argument("--prosody_dir", type=str, required=True,
                      help="Directory containing JSON feature files")
    parser.add_argument("--output_dir", type=str, required=True, 
                      help="Directory to save the model")
    parser.add_argument("--test_size", type=float, default=0.2, 
                      help="Fraction of data to use for validation")
    parser.add_argument("--random_state", type=int, default=42, 
                      help="Random seed for train/val split")
    parser.add_argument("--freeze_layers", type=str, default=None, 
                      help="Layers to freeze. Either a single number N to freeze first N layers, "
                           "or comma-separated list of specific layers (e.g., '0,1,2')")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                      help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, 
                      help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, 
                      help="Number of epochs")
    parser.add_argument("--patience", type=int, default=3, 
                      help="Early stopping patience")
    parser.add_argument("--save_total_limit", type=int, default=3, 
                      help="Number of best checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                      help="Path to a checkpoint directory to resume training from")
    parser.add_argument("--use_pca", action="store_true",
                             help="Use PCA for embeddings (default: False)")
    parser.add_argument("--pca_threshold", type=float, default=0.90,
                             help="PCA threshold for dataset (default: 0.50)")
    

    args = parser.parse_args()
    
    # Process freeze_layers argument
    if args.freeze_layers is not None:
        try:
            # Try to parse as comma-separated list first
            if ',' in args.freeze_layers:
                freeze_layers = [int(x.strip()) for x in args.freeze_layers.split(',')]
            else:
                # If not a list, treat as single number
                freeze_layers = int(args.freeze_layers)
        except ValueError:
            raise ValueError("freeze_layers must be either a number or comma-separated list of numbers")
    else:
        freeze_layers = None
    
    # Get list of story names (from audio files)
    audio_files = sorted([f for f in os.listdir(args.audio_dir) if f.endswith(".wav")])
    story_names = [f.replace(".wav", "") for f in audio_files]
    
    # Split stories into train and validation sets
    train_stories, val_stories = train_test_split(
        story_names, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    print(f"Training stories: {len(train_stories)}")
    print(f"Validation stories: {len(val_stories)}")
    
    # Save the splits for reproducibility
    split_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    
    with open(os.path.join(split_dir, "train_stories.txt"), 'w') as f:
        f.write('\n'.join(train_stories))
    with open(os.path.join(split_dir, "val_stories.txt"), 'w') as f:
        f.write('\n'.join(val_stories))
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    
    # Create datasets with story filtering
    train_dataset = ProsodyDataset(
        audio_dir=args.audio_dir, 
        prosody_dir=args.prosody_dir, 
        processor=processor,
        story_names=train_stories,
        use_pca=args.use_pca,
        pca_threshold=args.pca_threshold
    )

    val_dataset = ProsodyDataset(
        audio_dir=args.audio_dir, 
        prosody_dir=args.prosody_dir, 
        processor=processor,
        story_names=val_stories,
        use_pca=args.use_pca,
        pca_threshold=args.pca_threshold
    )
    
    print(f"Training set size (words): {len(train_dataset)}")
    print(f"Validation set size (words): {len(val_dataset)}")
    
    
    # Update the train_model call to include the resume_from_checkpoint argument
    train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_layers_to_freeze=freeze_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint
    )