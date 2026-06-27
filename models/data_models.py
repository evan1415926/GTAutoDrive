"""Shared data structures for perception and planning."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    """A single detected object in the driving scene."""
    class_name: str          # 'car', 'truck', 'person', 'motorcycle', 'bus', 'bicycle'
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords
    confidence: float        # 0-1
    center_x: float          # horizontal center normalized (0=left edge, 1=right edge)
    distance_est: float      # estimated distance (0=very near/bottom, 1=far/horizon)


@dataclass
class LaneInfo:
    """Lane detection result."""
    left_fit: np.ndarray | None = None    # 3 coefficients [a,b,c] for x = a*y² + b*y + c
    right_fit: np.ndarray | None = None   # 3 coefficients
    center_offset: float = 0.0            # vehicle offset from lane center (normalized by half lane width; <0=left, >0=right)
    curvature: float = 0.0                # lane curvature (1/radius in pixel space; <0=right turn, >0=left turn)
    lane_width: float = 0.0               # lane width in pixels at vehicle position
    confidence: float = 0.0               # 0-1 lane detection confidence
    detected: bool = False                # whether lanes were found


@dataclass
class ControlAction:
    """Output of the planner — key press/release flags."""
    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False
    reason: str = ''  # human-readable explanation
