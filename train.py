"""Training entry point for GTAutoDrive behavior cloning model.

Usage:
    python train.py                          # default paths
    python train.py --data data/recordings   # custom data dir
    python train.py --cpu                    # force CPU
    python train.py --epochs 50              # override epoch count
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from config.settings import AppConfig
from data.loader import load_data
from data.balancer import balance_classes
from model.network import create_model
from model.dataset import create_dataloaders
from model.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train GTAutoDrive behavior cloning model"
    )
    parser.add_argument("--data", type=str, default="data/recordings",
                        help="Path to recorded training data")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--balance-cap", type=int, default=500,
                        help="Max samples per class (default 500; 0 = no cap)")
    parser.add_argument("--include-recovery", action="store_true",
                        help="Also load recovery-mode frames")
    args = parser.parse_args()

    config = AppConfig()
    if args.cpu:
        config.train.device = "cpu"
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr

    print("=" * 50)
    print("GTAutoDrive — Model Training")
    print(f"Data dir:  {args.data}")
    print(f"Device:    {config.train.device}")
    print(f"Epochs:    {config.train.epochs}")
    print(f"Batch:     {config.train.batch_size}")
    print(f"LR:        {config.train.learning_rate}")
    print(f"Bal cap:   {args.balance_cap if args.balance_cap > 0 else 'equal'}")
    print("=" * 50)

    # Load (default: train mode only)
    print("\n[1/3] Loading data...")
    mode = None if args.include_recovery else 'train'
    frames, labels = load_data(args.data, mode=mode)

    # Balance
    print("\n[2/3] Balancing classes...")
    cap = args.balance_cap if args.balance_cap > 0 else None
    frames, labels = balance_classes(frames, labels, cap=cap)

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        frames, labels,
        batch_size=config.train.batch_size,
        val_split=config.train.val_split,
        num_workers=config.train.num_workers,
    )

    # Model
    print("\n[3/3] Training...")
    model = create_model(
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
    )
    trainer = Trainer(model, train_loader, val_loader, config.train)
    history = trainer.train()

    # Save
    model_path = Path(config.model.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    save_path = model_path / config.model.model_name
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Final metrics
    best_val_acc = max(history['val_acc'])
    print(f"Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
