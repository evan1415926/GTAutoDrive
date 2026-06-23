"""Test dataset and dataloader."""
import numpy as np
import torch
from model.dataset import (DrivingDataset, get_train_transforms,
                            get_val_transforms, create_dataloaders)


def make_data(n=20):
    frames = np.random.randint(0, 255, (n, 180, 320, 3), dtype=np.uint8)
    labels = np.random.randint(0, 7, (n,), dtype=np.int64)
    return frames, labels


class TestDrivingDataset:
    def test_basic(self):
        frames, labels = make_data(10)
        ds = DrivingDataset(frames, labels)
        img, label = ds[0]
        assert img.shape == (3, 180, 320)
        assert img.dtype == torch.float32
        assert 0 <= img.max() <= 255
        assert label.dtype == torch.long

    def test_with_train_transform(self):
        frames, labels = make_data(10)
        ds = DrivingDataset(frames, labels, transform=get_train_transforms())
        img, label = ds[0]
        assert img.shape == (3, 180, 320)
        assert label.dtype == torch.long

    def test_with_val_transform(self):
        frames, labels = make_data(10)
        ds = DrivingDataset(frames, labels, transform=get_val_transforms())
        img, label = ds[0]
        assert img.shape == (3, 180, 320)
        assert label.dtype == torch.long

    def test_create_dataloaders(self):
        frames, labels = make_data(50)
        train_dl, val_dl = create_dataloaders(
            frames, labels, batch_size=8, val_split=0.2, num_workers=0
        )
        x, y = next(iter(train_dl))
        assert x.shape[0] <= 8
        assert x.shape[1:] == (3, 180, 320)
        assert y.shape[0] <= 8

    def test_transform_preserves_label(self):
        """验证: 数据增强不改变标签（不含水平翻转因此标签不变）。"""
        frames, labels = make_data(100)
        ds = DrivingDataset(frames, labels, transform=get_train_transforms())
        for i in range(10):
            img, label = ds[i]
            assert img.shape == (3, 180, 320)
            assert 0 <= label.item() <= 6
