"""PyTorch Dataset and DataLoader for driving data."""
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.functional import (
    adjust_brightness, adjust_contrast, affine
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD).view(3, 1, 1)


class DrivingDataset(Dataset):
    """Dataset of (frame, label) pairs with optional transforms."""

    def __init__(self, frames: np.ndarray, labels: np.ndarray,
                 transform=None):
        self.frames = frames     # (N, H, W, 3) uint8 RGB
        self.labels = labels     # (N,) int64
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]   # (H, W, 3) uint8
        label = self.labels[idx]  # scalar int64

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: uint8 -> float32 tensor, [0, 255] range
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        label = torch.tensor(label, dtype=torch.long)
        return img, label


def get_train_transforms():
    """Training augmentations + ImageNet normalization.

    NOTE: No RandomHorizontalFlip because steering direction (A<->D, WA<->WD)
    would need label mirroring.
    """
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),   # uint8 -> float [0, 1]
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.RandomAffine(degrees=3, translate=(0.06, 0.0)),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """Validation: tensor conversion + ImageNet normalization."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),   # uint8 -> float [0, 1]
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Stacked (multi-frame) dataset ───────────────────────────────────────

class StackedDrivingDataset(Dataset):
    """Dataset of (stack, label) where stack = N consecutive same-label frames.

    Each item: (12, H, W) float32 tensor = 4 frames × 3 RGB channels,
    ImageNet-normalized per-frame.
    """

    def __init__(self, stacks: np.ndarray, labels: np.ndarray,
                 transform=None):
        self.stacks = stacks      # (N, 4, H, W, 3) uint8 RGB
        self.labels = labels      # (N,) int64
        self.transform = transform
        self._H = stacks.shape[2]
        self._W = stacks.shape[3]

    def __len__(self):
        return len(self.stacks)

    def __getitem__(self, idx):
        stack = self.stacks[idx]  # (4, H, W, 3) uint8
        label_val = self.labels[idx]

        # Convert 4 frames to float tensors: (4, 3, H, W) float [0, 1]
        frames = []
        for i in range(4):
            f = stack[i]
            f = torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
            frames.append(f)
        stacked = torch.stack(frames, dim=0)  # (4, 3, H, W)

        if self.transform is not None:
            stacked = self.transform(stacked)

        result = stacked.reshape(-1, self._H, self._W)  # (12, H, W)
        label = torch.tensor(label_val, dtype=torch.long)
        return result, label


class StackedTransform:
    """Apply same augmentation params to all 4 frames, then per-frame normalize.

    Uses functional transforms so all sub-frames get identical brightness,
    contrast, affine parameters — otherwise motion signal would be corrupted.
    """

    def __init__(self, training: bool = True):
        self.training = training

    def __call__(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: (4, 3, H, W) float [0, 1]
        if self.training:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            angle = random.uniform(-3.0, 3.0)
            tx = random.uniform(-0.06, 0.06) * stacked.shape[3]

            result = []
            for i in range(4):
                f = stacked[i]
                f = adjust_brightness(f, brightness)
                f = adjust_contrast(f, contrast)
                f = affine(f, angle=angle, translate=[tx, 0],
                          scale=1.0, shear=0.0)
                f = (f - _MEAN_T) / _STD_T
                result.append(f)
            return torch.cat(result, dim=0)  # (12, H, W)
        else:
            result = []
            for i in range(4):
                f = (stacked[i] - _MEAN_T) / _STD_T
                result.append(f)
            return torch.cat(result, dim=0)  # (12, H, W)


def get_stacked_train_transforms():
    return StackedTransform(training=True)


def get_stacked_val_transforms():
    return StackedTransform(training=False)


def create_dataloaders(frames: np.ndarray, labels: np.ndarray,
                       batch_size: int = 64, val_split: float = 0.2,
                       num_workers: int = 2):
    """Split data into train/val and return DataLoaders."""
    n = len(frames)
    n_val = int(n * val_split)
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = DrivingDataset(
        frames[train_idx], labels[train_idx],
        transform=get_train_transforms()
    )
    val_ds = DrivingDataset(
        frames[val_idx], labels[val_idx],
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_loader, val_loader


def create_stacked_dataloaders(stacks: np.ndarray, labels: np.ndarray,
                                batch_size: int = 64, val_split: float = 0.2,
                                num_workers: int = 2):
    """Split stacked data into train/val and return DataLoaders."""
    n = len(stacks)
    n_val = int(n * val_split)
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = StackedDrivingDataset(
        stacks[train_idx], labels[train_idx],
        transform=get_stacked_train_transforms()
    )
    val_ds = StackedDrivingDataset(
        stacks[val_idx], labels[val_idx],
        transform=get_stacked_val_transforms()
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    print(f"Train: {len(train_ds)} stacks, Val: {len(val_ds)} stacks")
    return train_loader, val_loader
