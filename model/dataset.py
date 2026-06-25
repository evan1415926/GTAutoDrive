"""PyTorch Dataset and DataLoader for driving data."""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_loader, val_loader
