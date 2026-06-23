"""Class-balanced downsampling to prevent model bias."""
import numpy as np
from config.settings import LABELS


def balance_classes(frames: np.ndarray,
                    labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample each class to the count of the smallest class.

    Args:
        frames: (N, H, W, 3) uint8
        labels: (N,) int64 class indices 0-6

    Returns:
        Balanced (frames, labels)
    """
    counts = np.bincount(labels, minlength=len(LABELS))
    print("Before balance:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {counts[i]}")

    min_count = counts[counts > 0].min()
    balanced_frames = []
    balanced_labels = []

    for class_idx in range(len(LABELS)):
        mask = labels == class_idx
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if len(indices) > min_count:
            indices = np.random.choice(indices, min_count, replace=False)
        balanced_frames.append(frames[indices])
        balanced_labels.append(labels[indices])

    result_frames = np.concatenate(balanced_frames, axis=0)
    result_labels = np.concatenate(balanced_labels, axis=0)

    # Shuffle
    perm = np.random.permutation(len(result_frames))
    result_frames = result_frames[perm]
    result_labels = result_labels[perm]

    print(f"After balance: {len(result_frames)} frames "
          f"({min_count} per class)")
    return result_frames, result_labels
