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


def _collect_frame_metadata(search_root: Path) -> list[dict]:
    """Scan label subdirs, collect all frame files, sort by mtime.

    Uses file modification time to reconstruct global temporal order
    (counter resets per recording session, so mtime is authoritative).
    Returns list of {mtime, path, label, label_idx} sorted by mtime.
    """
    import re
    pattern = re.compile(r'frame_(\d+)\.png$')
    records = []
    for label_dir in sorted(search_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in LABEL_TO_IDX:
            continue
        for png in label_dir.glob('frame_*.png'):
            m = pattern.search(png.name)
            if not m:
                continue
            records.append({
                'mtime': png.stat().st_mtime,
                'path': png,
                'label': label,
                'label_idx': LABEL_TO_IDX[label],
            })
    records.sort(key=lambda r: r['mtime'])
    return records


def load_data_stacked(data_dir: str, mode: str | None = 'train',
                      stack_size: int = 4, stride: int = 1
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Load frames in temporal order, assemble N-frame history stacks.

    Sorts frames by file modification time to reconstruct capture order
    (counters reset per recording session). Each stack is a sliding window
    of ``stack_size`` consecutive frames; the label comes from the LAST
    frame in the window (standard behavior-cloning-with-history approach).

    Args:
        data_dir: root directory (e.g. 'data/recordings')
        mode: 'train' / 'recovery' / None
        stack_size: frames per stack (default 4 → 12-channel input)
        stride: frames to advance after each stack (default 1, overlapping)

    Returns:
        stacks: (N, stack_size, H, W, 3) uint8 RGB
        labels: (N,) int64 class indices
    """
    base = Path(data_dir)
    if mode is not None and (base / mode).is_dir():
        search_root = base / mode
    else:
        search_root = base

    records = _collect_frame_metadata(search_root)
    if not records:
        raise FileNotFoundError(
            f"No frame PNGs found in {search_root}. "
            f"Expected subdirs: {LABELS}"
        )

    stacks = []
    labels = []
    frames_skipped = 0

    i = 0
    while i + stack_size <= len(records):
        sub_frames = []
        for j in range(stack_size):
            img_bgr = cv2.imread(str(records[i + j]['path']))
            if img_bgr is None:
                print(f"[WARN] Corrupt file: {records[i + j]['path']}")
                break
            sub_frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if len(sub_frames) == stack_size:
            stacks.append(np.stack(sub_frames, axis=0))
            labels.append(records[i + stack_size - 1]['label_idx'])
        else:
            frames_skipped += stack_size

        i += stride

    if not stacks:
        raise RuntimeError(
            f"No valid {stack_size}-frame stacks could be assembled."
        )

    result_stacks = np.stack(stacks, axis=0)
    result_labels = np.array(labels, dtype=np.int64)
    print(f"Loaded {result_stacks.shape[0]} stacks "
          f"from {len(records)} frames "
          f"(stack_size={stack_size}, stride={stride})")
    return result_stacks, result_labels


def load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load frames and labels from a .npz file."""
    data = np.load(path)
    return data["frames"], data["labels"]


def save_batch(frames: np.ndarray, labels: np.ndarray, path: str):
    """Save frames and labels as compressed .npz."""
    np.savez_compressed(path, frames=frames, labels=labels)
    print(f"Saved {len(frames)} frames to {path}")
