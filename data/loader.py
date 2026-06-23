"""Load recorded training data from PNG files on disk."""
from pathlib import Path
import numpy as np
import cv2
from config.settings import LABELS, LABEL_TO_IDX


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Scan data_dir for label subdirs, load all PNGs.

    Returns:
        frames: (N, H, W, 3) uint8 RGB array
        labels: (N,) int64 array of class indices (0-6)
    """
    frames_list = []
    labels_list = []

    for label in LABELS:
        label_dir = Path(data_dir) / label
        if not label_dir.is_dir():
            continue
        for png_file in sorted(label_dir.glob("frame_*.png")):
            # OpenCV reads as BGR, convert to RGB
            img_bgr = cv2.imread(str(png_file))
            if img_bgr is None:
                print(f"[WARN] Corrupt file: {png_file}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames_list.append(img_rgb)
            labels_list.append(LABEL_TO_IDX[label])

    if not frames_list:
        raise FileNotFoundError(
            f"No frame PNGs found in {data_dir}. "
            f"Expected subdirs: {LABELS}"
        )

    frames = np.stack(frames_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)
    print(f"Loaded {len(frames)} frames, {len(labels)} labels "
          f"from {data_dir}")
    return frames, labels


def load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load frames and labels from a .npz file."""
    data = np.load(path)
    return data["frames"], data["labels"]


def save_batch(frames: np.ndarray, labels: np.ndarray, path: str):
    """Save frames and labels as compressed .npz."""
    np.savez_compressed(path, frames=frames, labels=labels)
    print(f"Saved {len(frames)} frames to {path}")
