"""Screen capture via mss (cross-platform screenshot using GDI)."""
import cv2
import numpy as np
import mss
from config.settings import CaptureConfig


class ScreenCapture:
    def __init__(self, config: CaptureConfig):
        self._monitor = config.monitor_index
        self._target_w = config.capture_width
        self._target_h = config.capture_height
        self._fps = config.target_fps
        self._sct = None

    def open(self) -> bool:
        try:
            self._sct = mss.MSS()
            if self._monitor >= len(self._sct.monitors):
                print(f"[CAPTURE] Monitor {self._monitor} unavailable. "
                      f"Available: {len(self._sct.monitors)}")
                return False
            monitor = self._sct.monitors[self._monitor]
            print(f"[CAPTURE] Monitor {self._monitor} "
                  f"({monitor['width']}x{monitor['height']}) ready.")
            return True
        except Exception as e:
            print(f"[CAPTURE] Error opening capture: {e}")
            return False

    def read(self) -> np.ndarray | None:
        if self._sct is None:
            return None
        img = self._sct.grab(self._sct.monitors[self._monitor])
        frame = np.array(img)[:, :, :3]  # BGRA -> BGR
        if frame.shape[1] != self._target_w or frame.shape[0] != self._target_h:
            frame = cv2.resize(frame, (self._target_w, self._target_h))
        return frame

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._target_w, self._target_h)

    def close(self):
        if self._sct is not None:
            self._sct.close()
            self._sct = None
