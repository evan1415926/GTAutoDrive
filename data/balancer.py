"""Class-balanced downsampling to prevent model bias."""
import numpy as np
from config.settings import LABELS


def balance_classes(frames: np.ndarray,
                    labels: np.ndarray,
                    cap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Downsample each class to keep the dataset balanced.

    By default, every class is capped at the count of the smallest class
    (strict balance). Pass ``cap`` to set a maximum per class instead —
    classes below the cap are kept in full.

    Args:
        frames: (N, H, W, 3) uint8
        labels: (N,) int64 class indices 0-6
        cap: maximum samples per class. If None, uses min class count.

    Returns:
        Balanced (frames, labels)
    """
    counts = np.bincount(labels, minlength=len(LABELS))
    print("Before balance:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {counts[i]}")

    max_count = cap if cap is not None else counts[counts > 0].min()
    balanced_frames = []
    balanced_labels = []

    for class_idx in range(len(LABELS)):
        mask = labels == class_idx
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if len(indices) > max_count:
            indices = np.random.choice(indices, max_count, replace=False)
        balanced_frames.append(frames[indices])
        balanced_labels.append(labels[indices])

    result_frames = np.concatenate(balanced_frames, axis=0)
    result_labels = np.concatenate(balanced_labels, axis=0)

    # Shuffle
    perm = np.random.permutation(len(result_frames))
    result_frames = result_frames[perm]
    result_labels = result_labels[perm]

    print(f"After balance: {len(result_frames)} frames "
          f"(max {max_count} per class)")
    return result_frames, result_labels
