"""Debug overlay drawing on the main frame."""
import cv2
import numpy as np
from config.settings import LABELS


def draw_overlay(frame_bgr: np.ndarray, action_label: str, fps: float,
                 probs: np.ndarray | None = None) -> np.ndarray:
    """Draw status bar and action info on frame.

    Args:
        frame_bgr: BGR frame, will be modified in-place
        action_label: current action string (e.g. 'W', 'WA')
        fps: current FPS
        probs: (7,) probability array, or None

    Returns:
        modified frame
    """
    h, w = frame_bgr.shape[:2]

    # Semi-transparent status bar at top
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(frame_bgr, 0.5, overlay, 0.5, 0)

    # FPS
    cv2.putText(frame_bgr, f"FPS: {fps:.0f}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Action
    action_colors = {
        'W': (0, 255, 0), 'WA': (255, 200, 0), 'WD': (0, 200, 255),
        'A': (255, 128, 0), 'D': (255, 0, 128), 'S': (0, 0, 255),
        'NONE': (128, 128, 128),
    }
    color = action_colors.get(action_label, (255, 255, 255))
    cv2.putText(frame_bgr, f"ACT: {action_label}", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Probability bars
    if probs is not None and len(probs) == 7:
        bar_start_y = 52
        bar_h = 12
        bar_gap = 2
        for i, (label, prob) in enumerate(zip(LABELS, probs)):
            y = bar_start_y + i * (bar_h + bar_gap)
            bw = int(prob * 120)
            cv2.rectangle(frame_bgr, (120, y), (120 + bw, y + bar_h),
                          color, -1)
            cv2.putText(frame_bgr, f"{label}", (90, y + bar_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    return frame_bgr
