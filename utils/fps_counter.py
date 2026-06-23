"""Rolling-average FPS counter."""
import time
import collections


class FPSCounter:
    def __init__(self, window: int = 30):
        self._window = window
        self._times = collections.deque(maxlen=window)
        self._last = time.perf_counter()

    def update(self):
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now

    def get(self) -> float:
        if not self._times:
            return 0.0
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0

    def reset(self):
        self._times.clear()
        self._last = time.perf_counter()
