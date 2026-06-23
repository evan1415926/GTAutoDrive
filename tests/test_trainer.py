"""Smoke test: run one epoch on synthetic data."""
import torch
import numpy as np
from model.trainer import Trainer
from model.network import create_model
from model.dataset import DrivingDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
from config.settings import TrainConfig


class TestTrainer:
    def test_one_epoch_no_crash(self):
        # Synthetic data: 100 frames, balanced 7 classes
        n = 100
        frames = np.random.randint(0, 255, (n, 180, 320, 3), dtype=np.uint8)
        labels = np.tile(np.arange(7), (n // 7) + 1)[:n].astype(np.int64)

        train_ds = DrivingDataset(frames[:80], labels[:80],
                                  transform=get_train_transforms())
        val_ds = DrivingDataset(frames[80:], labels[80:],
                                transform=get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        model = create_model(num_classes=7)
        cfg = TrainConfig(epochs=2, early_stop_patience=10, device="cpu")
        trainer = Trainer(model, train_loader, val_loader, cfg)
        history = trainer.train()

        assert len(history['train_loss']) >= 1
        assert history['train_loss'][0] > 0  # loss should be positive
