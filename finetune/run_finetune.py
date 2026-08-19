"""
Fine-tune a self-supervised speech model on per-TR eGeMAPS prosody targets.

Stage 1 of the pipeline. The resulting encoder is what `extract/wav2vec.py`
then runs over the stimuli to produce the audio band for the encoding models.

Prerequisites
-------------
    python -m prep.make_story_splits        # train / val / held-out stories
    python -m prep.make_finetune_targets    # 88 eGeMAPS functionals per TR

Examples
--------
Recommended 24-layer arm, freezing the bottom half::

    python -m finetune.run_finetune --model wav2vec2-robust

The emotion arm (12 layers; `auto` resolves the freeze depth per model)::

    python -m finetune.run_finetune --model emotion

Layer-wise LR decay instead of a hard freeze boundary::

    python -m finetune.run_finetune --model wav2vec2-robust \\
        --freeze-layers none --llrd 0.9

Always keep the no-fine-tuning frozen baseline as the control — see
`extract/wav2vec.py`. If it wins, fine-tuning is hurting.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoFeatureExtractor

from config import (EGEMAPS_N_FUNCTIONALS, FINETUNE_OUT, FINETUNE_SPLIT,
                    FINETUNE_TARGET_DIR, HELD_OUT_STORY, STIMULI_16K_DIR)
from . import REGISTRY, format_registry, resolve_model
from .dataset import ProsodyDataset
from .training import train_model


def parse_freeze_layers(value: str):
    """'12' -> 12 ; '0,1,2' -> [0,1,2] ; 'none' -> None ; 'auto' -> 'auto'."""
    if value is None or value.lower() == "none":
        return None
    if value.lower() == "auto":
        # Resolved against the chosen model's depth once it is known.
        return "auto"
    try:
        if "," in value:
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--freeze-layers takes an integer, a comma-separated list, or 'none'"
        )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    paths = p.add_argument_group("paths")
    paths.add_argument("--audio-dir", default=str(STIMULI_16K_DIR),
                       help="directory of 16 kHz story wavs")
    paths.add_argument("--target-dir", default=str(FINETUNE_TARGET_DIR),
                       help="output of prep/make_finetune_targets.py")
    paths.add_argument("--split", default=str(FINETUNE_SPLIT),
                       help="stories_split.json from prep/make_story_splits.py")
    paths.add_argument("--output-dir", default=str(FINETUNE_OUT))

    task = p.add_argument_group("task")
    task.add_argument("--held-out-story", default=HELD_OUT_STORY,
                      help="story reserved for the encoding test; fine-tuning "
                           "aborts if it appears in the split")
    task.add_argument("--trim", type=int, default=5,
                      help="must match the TRIM used to build the targets")
    task.add_argument("--expect-n-features", type=int,
                      default=EGEMAPS_N_FUNCTIONALS,
                      help=f"fail unless the prosody targets have exactly this "
                           f"many features (default {EGEMAPS_N_FUNCTIONALS}, the "
                           f"eGeMAPSv02 functionals). Pass 0 to disable.")

    model = p.add_argument_group("model")
    model.add_argument("--model", "--model-type", dest="model",
                       default="wav2vec2", metavar="NAME",
                       help="registry key (" + ", ".join(sorted(REGISTRY)) +
                            ") or any Hugging Face id / local directory. "
                            "Run --list-models for what each one is.")
    model.add_argument("--base-model", default=None,
                       help="explicit checkpoint, overriding --model")
    model.add_argument("--freeze-layers", type=parse_freeze_layers,
                       default="auto",
                       help="N (freeze the first N), a list like '0,1,2', "
                            "'none', or 'auto' to use the model's registry "
                            "default. The CNN front end is always frozen.")
    model.add_argument("--truncate-layers", type=int, default=None,
                       metavar="N",
                       help="keep only the first N transformer layers. Use "
                            "'--model wav2vec2-robust --truncate-layers 12' to "
                            "build a depth- and base-matched control for the "
                            "12-layer emotion model.")
    model.add_argument("--list-models", action="store_true",
                       help="print the model registry and exit")

    train = p.add_argument_group("training")
    train.add_argument("--learning-rate", type=float, default=3e-5)
    train.add_argument("--llrd", type=float, default=None, metavar="DECAY",
                       help="layer-wise learning-rate decay in (0, 1], e.g. "
                            "0.9: lower layers train proportionally slower. A "
                            "gentler alternative to freezing; combine with "
                            "--freeze-layers none to keep every layer plastic "
                            "but stable.")
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--num-epochs", type=int, default=10)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--metric-for-best", default="eval_loss",
                       choices=["eval_loss", "eval_mean_r", "eval_mean_r2"],
                       help="checkpoint selection metric. eval_loss is MSE, "
                            "which a model can minimise by predicting each "
                            "feature's mean; eval_mean_r cannot be gamed that "
                            "way and is the better choice if that shows up.")
    train.add_argument("--save-total-limit", type=int, default=3)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--resume-from-checkpoint", default=None)
    train.add_argument("--torch-compile", action="store_true")

    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.list_models:
        print("Available models\n")
        print(format_registry())
        print("\nAny other value is passed through as a Hugging Face id or a "
              "local directory.")
        return

    spec = resolve_model(args.model)

    # Resolve 'auto' against this model's depth. The emotion checkpoint has 12
    # layers, so a value tuned for a 24-layer backbone would be wrong.
    freeze_layers = args.freeze_layers
    if freeze_layers == "auto":
        # Resolve against the depth the model will actually have: truncation
        # happens before freezing, so using the untruncated depth here would
        # freeze the entire remaining stack and train nothing but the heads.
        depth = args.truncate_layers or spec.n_layers
        if depth is None:
            raise SystemExit(
                f"--freeze-layers auto needs a known depth, and {args.model!r} "
                f"is not in the registry. Pass an explicit value."
            )
        freeze_layers = depth // 2
        print(f"--freeze-layers auto -> {freeze_layers} "
              f"(bottom half of {depth} layers"
              f"{', after truncation' if args.truncate_layers else f' in {spec.key}'})")

    split_path = Path(args.split)
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found — run `python -m prep.make_story_splits` first"
        )
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)

    train_stories, val_stories = split["train"], split["val"]
    overlap = set(train_stories) & set(val_stories)
    if overlap:
        raise ValueError(f"train/val overlap in {split_path}: {sorted(overlap)}")

    # Hard guarantee: fine-tuning must never see the story the encoding models
    # are tested on. If it did, the encoder would have been optimised on the
    # audio of the test story, and every downstream encoding score on it
    # would be contaminated.
    held_out = set(split.get("held_out_test") or []) | {args.held_out_story}
    leaked = held_out & (set(train_stories) | set(val_stories))
    if leaked:
        raise ValueError(
            f"Held-out story/stories {sorted(leaked)} appear in the "
            f"fine-tuning split. Regenerate it with "
            f"`python -m prep.make_story_splits`; fine-tuning on the encoding "
            f"test story invalidates every downstream encoding result."
        )

    print(f"Train stories: {len(train_stories)}")
    print(f"Val stories  : {len(val_stories)} -> {val_stories}")
    print(f"Held out     : {sorted(held_out)} — verified absent from train and val")

    base_model_name = args.base_model or spec.checkpoint
    print(f"\nModel  : {spec.key} -> {base_model_name}")
    if spec.note:
        print(f"         {spec.note}")
    if args.truncate_layers:
        print(f"         truncating to the first {args.truncate_layers} layers")
    print(f"Loading processor: {base_model_name}")
    processor = AutoFeatureExtractor.from_pretrained(base_model_name)

    common = dict(
        audio_dir=args.audio_dir,
        target_dir=args.target_dir,
        processor=processor,
        trim=args.trim,
        expect_n_features=args.expect_n_features or None,
    )

    print("\nBuilding training dataset ...")
    train_dataset = ProsodyDataset(**common, story_names=train_stories)

    print("Building validation dataset (reusing the training scalers) ...")
    val_dataset = ProsodyDataset(
        **common, story_names=val_stories,
        scalers=train_dataset.get_fitted_scalers(),
    )

    print(f"\nTrain : {len(train_dataset):,} windows")
    print(f"Val   : {len(val_dataset):,} windows")
    print(f"Labels: {train_dataset.label_dim} prosody features")
    print(f"        (eGeMAPSv02 functionals: "
          f"{train_dataset.feature_names[:3]} ... "
          f"{train_dataset.feature_names[-1]})\n")

    train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        model_type=args.model,
        # The *resolved* checkpoint, not args.base_model. Passing the raw
        # (usually None) override meant train_model re-resolved through the
        # registry and rejected any plain Hugging Face id or local directory —
        # after both datasets had already been built.
        base_model_name=base_model_name,
        num_layers_to_freeze=freeze_layers,
        truncate_layers=args.truncate_layers,
        llrd=args.llrd,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.num_epochs,
        patience=args.patience,
        metric_for_best=args.metric_for_best,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        torch_compile=args.torch_compile,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
