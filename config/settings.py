"""All configuration and shared constants for GTAutoDrive."""
from dataclasses import dataclass, field


# ── Shared label constants ────────────────────────────────────────────

LABELS = ['W', 'WA', 'WD', 'A', 'D', 'S', 'NONE']
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

LABEL_TO_KEYS = {
    'W':    {'w': True,  'a': False, 's': False, 'd': False},
    'WA':   {'w': True,  'a': True,  's': False, 'd': False},
    'WD':   {'w': True,  'a': False, 's': False, 'd': True},
    'A':    {'w': False, 'a': True,  's': False, 'd': False},
    'D':    {'w': False, 'a': False, 's': False, 'd': True},
    'S':    {'w': False, 'a': False, 's': True,  'd': False},
    'NONE': {'w': False, 'a': False, 's': False, 'd': False},
}


def keys_to_label(w: bool, a: bool, s: bool, d: bool) -> str:
    """Convert WASD boolean state to a 7-class label string."""
    if s:
        return 'S'
    if w and a:
        return 'WA'
    if w and d:
        return 'WD'
    if w:
        return 'W'
    if a:
        return 'A'
    if d:
        return 'D'
    return 'NONE'


# ── Configuration dataclasses ─────────────────────────────────────────

@dataclass
class CaptureConfig:
    monitor_index: int = 1          # mss monitor index (1=primary, 0=all)
    target_fps: int = 30
    capture_width: int = 1920
    capture_height: int = 1080


@dataclass
class DataConfig:
    frame_width: int = 320
    frame_height: int = 180
    fps: int = 10
    save_dir: str = "data/recordings"


@dataclass
class ModelConfig:
    num_classes: int = 7
    dropout: float = 0.5
    model_dir: str = "data/models"
    model_name: str = "driving_model.pt"


@dataclass
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 5e-4
    epochs: int = 100
    early_stop_patience: int = 15
    lr_patience: int = 5
    val_split: float = 0.2
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class InferenceConfig:
    ema_alpha: float = 0.15         # probs = α*new + (1-α)*prev (lower = smoother)
    confidence_threshold: float = 0.2  # NONE if max prob < threshold
    w_bias: float = 0.5             # added to W logit (positive = prefer forward)
    target_fps: int = 30


@dataclass
class PerceptionConfig:
    detection_conf: float = 0.35
    detection_size: tuple = (640, 384)
    lane_roi_top: float = 0.55
    lane_n_windows: int = 9
    lane_window_margin: int = 80
    lane_min_pixels: int = 50
    # Perspective transform tuning (in ROI coordinates, as fraction)
    lane_persp_top_y: float = 0.40       # vanishing point Y in ROI (0=top of ROI)
    lane_persp_margin_top: float = 0.15  # horizontal margin at top of trapezoid
    lane_persp_margin_bot: float = 0.10  # horizontal margin at bottom of trapezoid


@dataclass
class PlannerConfig:
    emergency_brake_dist: float = 0.10
    obstacle_brake_dist: float = 0.22
    lane_offset_thresh: float = 0.25
    steer_gain: float = 2.5
    curve_steer_gain: float = 0.8


@dataclass
class KeysConfig:
    throttle: str = "w"
    brake: str = "s"
    steer_left: str = "a"
    steer_right: str = "d"
    panic_key: str = "f8"


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    keys: KeysConfig = field(default_factory=KeysConfig)
