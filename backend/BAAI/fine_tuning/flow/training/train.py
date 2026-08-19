"""
train.py

Entry point for Phase 2. 
Added: Progress bar with ETA (Estimated Time Remaining)
"""

import argparse
import os
import time
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from fine_tuning.flow.training.dataset_loader import get_dataloader
from fine_tuning.flow.training.optimizer import build_optimizer_and_scheduler
from fine_tuning.flow.training.trainer import Trainer
from fine_tuning.flow.training.evaluator import evaluate, print_eval_result

BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # Smaller & faster model
MAX_SEQ_LENGTH = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BGE for Skill Matching.")

    parser.add_argument("--train-csv", default="datasets/train_pairs.csv")
    parser.add_argument("--valid-csv", default="datasets/test_pairs.csv")
    parser.add_argument("--model-name", default=BASE_MODEL_NAME)
    parser.add_argument("--output-dir", default="fine_tuning/flow/saved_model/bge_finetuned")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"{args.train_csv} not found. Run pair_builder.py first.")

    print("Loading tokenizer + dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_loader = get_dataloader(
        args.train_csv,
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )
    valid_loader = get_dataloader(
        args.valid_csv,
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    print(f"Loading base model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name).to(device)

    num_training_steps = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        num_training_steps=num_training_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        pooling=args.pooling,
        margin=args.margin,
        output_dir=args.output_dir,
    )

    print(f"Starting training: {args.epochs} epochs | Batch size: {args.batch_size}")
    print("=" * 70)

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        trainer.fit(
            train_loader,
            valid_loader,
            num_epochs=1,                    # Train one epoch at a time
            tokenizer=tokenizer,
            save_every_epoch=args.save_every_epoch,
        )

        epoch_time = time.time() - epoch_start
        total_time = time.time() - total_start_time
        avg_epoch_time = total_time / epoch
        remaining_epochs = args.epochs - epoch
        eta = remaining_epochs * avg_epoch_time

        print(f"Epoch {epoch} completed in {epoch_time/60:.1f} minutes")
        print(f"Estimated time remaining: {eta/3600:.1f} hours")

    total_time = time.time() - total_start_time
    print(f"\n✅ Training Completed in {total_time/3600:.1f} hours!")

    print("Running final evaluation on validation set...")
    result = evaluate(model, valid_loader, device=device, pooling=args.pooling)
    print_eval_result(result, split_name="validation (final)")

    print(f"Best model saved to: {os.path.join(args.output_dir, 'best')}")
    print(f"Final model saved to: {os.path.join(args.output_dir, 'final')}")


if __name__ == "__main__":
    main()