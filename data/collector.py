"""Data collector: record (frame, action) pairs for behavior cloning.

Frames saved as PNG files organized by mode + label class:
  data/recordings/train/W/frame_00001.png
  data/recordings/recovery/W/frame_00002.png
"""
import time
from pathlib import Path
import cv2
import numpy as np
from config.settings import DataConfig, keys_to_label


class DataCollector:
    def __init__(self, config: DataConfig):
        self.save_dir = Path(config.save_dir)
        self.fw = config.frame_width
        self.fh = config.frame_height
        self._recording = False
        self._mode = 'train'       # 'train' or 'recovery'
        self._count = 0
        self._start_time = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def frame_count(self) -> int:
        return self._count

    def start(self):
        self._recording = True
        self._count = 0
        self._start_time = time.perf_counter()
        print(f"[REC] Started ({self._mode} mode)")

    def stop(self):
        self._recording = False
        elapsed = time.perf_counter() - self._start_time
        print(f"[REC] Stopped. {self._count} frames in {elapsed:.1f}s "
              f"({self._count / max(elapsed, 1):.1f} FPS)")

    def toggle_mode(self):
        self._mode = 'recovery' if self._mode == 'train' else 'train'
        print(f"[REC] Mode: {self._mode}")

    def capture(self, frame_bgr: np.ndarray, w: bool, a: bool,
                s: bool, d: bool):
        """Record one frame with current key state."""
        if not self._recording:
            return

        label = keys_to_label(w, a, s, d)
        label_dir = self.save_dir / self._mode / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Resize frame (keep BGR — cv2.imwrite expects BGR)
        small = cv2.resize(frame_bgr, (self.fw, self.fh))
        filename = label_dir / f"frame_{self._count:05d}.png"
        cv2.imwrite(str(filename), small)
        self._count += 1

    def flush(self):
        """No-op: files are already written to disk."""
        pass
