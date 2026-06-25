"""Load recorded training data from PNG files on disk.

Directory layout (new):
  data/recordings/train/W/frame_*.png
  data/recordings/recovery/W/frame_*.png

Old layout (no mode subdir) is still auto-detected.
"""
from pathlib import Path
import numpy as np
import cv2
from config.settings import LABELS, LABEL_TO_IDX


def load_data(data_dir: str, mode: str | None = 'train') -> tuple[np.ndarray, np.ndarray]:
    """Scan data_dir, load PNGs filtered by recording mode.

    Args:
        data_dir: root directory (e.g. 'data/recordings')
        mode: 'train' / 'recovery' / None (load all, old format compat)

    Returns:
        frames: (N, H, W, 3) uint8 RGB array
        labels: (N,) int64 array of class indices (0-6)
    """
    base = Path(data_dir)

    # Auto-detect: new layout has train/ subdir
    if mode is not None and (base / mode).is_dir():
        search_root = base / mode
    else:
        search_root = base

    frames_list = []
    labels_list = []

    for label in LABELS:
        label_dir = search_root / label
        if not label_dir.is_dir():
            continue
        for png_file in sorted(label_dir.glob("frame_*.png")):
            img_bgr = cv2.imread(str(png_file))
            if img_bgr is None:
                print(f"[WARN] Corrupt file: {png_file}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames_list.append(img_rgb)
            labels_list.append(LABEL_TO_IDX[label])

    if not frames_list:
        raise FileNotFoundError(
            f"No frame PNGs found in {search_root}. "
            f"Expected subdirs: {LABELS}"
        )

    frames = np.stack(frames_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)
    print(f"Loaded {len(frames)} frames, {len(labels)} labels "
          f"from {search_root}")
    return frames, labels


def load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load frames and labels from a .npz file."""
    data = np.load(path)
    return data["frames"], data["labels"]


def save_batch(frames: np.ndarray, labels: np.ndarray, path: str):
    """Save frames and labels as compressed .npz."""
    np.savez_compressed(path, frames=frames, labels=labels)
    print(f"Saved {len(frames)} frames to {path}")
