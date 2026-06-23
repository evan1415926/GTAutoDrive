"""Screen capture via dxcam (DXGI desktop duplication)."""
import cv2
import numpy as np
from config.settings import CaptureConfig


class ScreenCapture:
    def __init__(self, config: CaptureConfig):
        self._monitor = config.monitor_index
        self._target_w = config.capture_width
        self._target_h = config.capture_height
        self._fps = config.target_fps
        self._camera = None

    def open(self) -> bool:
        import dxcam
        try:
            self._camera = dxcam.create(
                output_idx=self._monitor,
                output_color="BGR",
            )
            if self._camera is None:
                print(f"[CAPTURE] Monitor {self._monitor} unavailable.")
                return False
            # Warm up
            test = self._camera.grab()
            if test is None:
                print("[CAPTURE] Grab returned None — try running GTA in "
                      "borderless window mode.")
                return False
            print(f"[CAPTURE] Monitor {self._monitor} "
                  f"({test.shape[1]}x{test.shape[0]}) ready.")
            return True
        except Exception as e:
            print(f"[CAPTURE] Error opening capture: {e}")
            return False

    def read(self) -> np.ndarray | None:
        if self._camera is None:
            return None
        frame = self._camera.grab()
        if frame is not None and (frame.shape[1] != self._target_w
                                  or frame.shape[0] != self._target_h):
            frame = cv2.resize(frame, (self._target_w, self._target_h))
        return frame

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._target_w, self._target_h)

    def close(self):
        if self._camera is not None:
            del self._camera
            self._camera = None
