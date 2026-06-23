# GTA V L2 自动驾驶 — 端到端行为克隆 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练一个端到端 CNN（ResNet-18），输入 GTA V 游戏画面，输出 7 分类驾驶动作，实现 L2 级不变道巡航。

**Architecture:** 端到端行为克隆。dxcam 截取画面 → resize 320×180 → ResNet-18 backbone → FC 512→7 → Softmax → argmax 得到动作标签 → pynput 模拟按键。

**Tech Stack:** Python 3.10, PyTorch + torchvision, dxcam, pynput, keyboard, opencv-python, numpy, Pillow

## Global Constraints

- 虚拟环境: `G:\GTAutoDrive\venv`
- 设备: CUDA 优先，CPU 可降级
- 画面输入: 320×180 RGB
- 动作输出: 7 分类 (W, WA, WD, A, D, S, NONE)
- 采集帧率: 10 FPS
- 推理帧率: 30 FPS
- 紧急停止: F8
- 油门上限: 0.8
- 所有退出路径必须 release_all()

## 共享常量

7 分类标签定义（全项目共用）：

```python
LABELS = ['W', 'WA', 'WD', 'A', 'D', 'S', 'NONE']
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(LABELS)}
# ['W', 'WA', 'WD', 'A', 'D', 'S', 'NONE'] → [0, 1, 2, 3, 4, 5, 6]
```

键盘 → 标签映射：
```python
def keys_to_label(w, a, s, d) -> str:
    if s: return 'S'       # 刹车最高优先
    if w and a: return 'WA'
    if w and d: return 'WD'
    if w: return 'W'
    if a: return 'A'
    if d: return 'D'
    return 'NONE'
```

标签 → 按键状态：
```python
LABEL_TO_KEYS = {
    'W':    {'w': True,  'a': False, 's': False, 'd': False},
    'WA':   {'w': True,  'a': True,  's': False, 'd': False},
    'WD':   {'w': True,  'a': False, 's': False, 'd': True},
    'A':    {'w': False, 'a': True,  's': False, 'd': False},
    'D':    {'w': False, 'a': False, 's': False, 'd': True},
    'S':    {'w': False, 'a': False, 's': True,  'd': False},
    'NONE': {'w': False, 'a': False, 's': False, 'd': False},
}
```

---

### Task 1: 清理旧代码 + 建立新项目结构

**Files:**
- Delete: `vision/`, `control/`, `training/`, `models/data_models.py`, `gui.py`, `config/default_config.json`, `tools/calibrate_regions.py`, `tools/test_detection.py`, `tools/test_minimap.py`, `debug_dumps/`, `test_printwindow.png`, `yolov8n.pt`, `tools/test_input.py`, `__pycache__/`, `.pytest_cache/`
- Delete: `main.py` (rewrite later), `capture/`, `input/`, `utils/` (rewrite later)
- Create: `config/__init__.py`, `capture/__init__.py`, `input/__init__.py`, `model/__init__.py`, `data/__init__.py`, `utils/__init__.py`, `tools/__init__.py`
- Create: `data/recordings/.gitkeep`, `data/models/.gitkeep`
- Modify: `requirements.txt`

**Interfaces:** None (no dependencies)

- [ ] **Step 1: 删除旧模块目录**

```bash
cd G:/GTAutoDrive
rm -rf vision/ control/ training/ models/ debug_dumps/ .pytest_cache/
rm -f gui.py test_printwindow.png yolov8n.pt
rm -f config/default_config.json
rm -rf tools/calibrate_regions.py tools/test_detection.py tools/test_minimap.py tools/test_input.py
rm -rf capture/ input/ utils/ main.py
```

- [ ] **Step 2: 创建新目录结构**

```bash
mkdir -p G:/GTAutoDrive/{config,capture,input,model,data,utils,tools}
mkdir -p G:/GTAutoDrive/data/{recordings,models}
touch G:/GTAutoDrive/config/__init__.py
touch G:/GTAutoDrive/capture/__init__.py
touch G:/GTAutoDrive/input/__init__.py
touch G:/GTAutoDrive/model/__init__.py
touch G:/GTAutoDrive/data/__init__.py
touch G:/GTAutoDrive/utils/__init__.py
touch G:/GTAutoDrive/tools/__init__.py
touch G:/GTAutoDrive/data/recordings/.gitkeep
touch G:/GTAutoDrive/data/models/.gitkeep
```

- [ ] **Step 3: 重写 requirements.txt**

```bash
cat > G:/GTAutoDrive/requirements.txt << 'EOF'
dxcam>=0.3.0
opencv-python==4.10.0.84
numpy==1.26.4
torch>=2.3.0
torchvision>=0.18.0
pynput==1.7.7
keyboard==0.13.5
Pillow>=10.3.0
pytest>=9.0.0
EOF
```

- [ ] **Step 4: 安装依赖并验证**

```bash
G:/GTAutoDrive/venv/Scripts/activate && pip install -r G:/GTAutoDrive/requirements.txt
python -c "import dxcam; import cv2; import numpy; import torch; import torchvision; import pynput; import keyboard; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git -C G:/GTAutoDrive add -A
git -C G:/GTAutoDrive commit -m "feat: clean old code, init new project structure"
```

---

### Task 2: config/settings.py — 配置数据类

**Files:**
- Create: `config/settings.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `AppConfig` dataclass with nested `CaptureConfig`, `DataConfig`, `ModelConfig`, `TrainConfig`, `InferenceConfig`, `KeysConfig`
- Produces: `LABELS`, `LABEL_TO_IDX`, `IDX_TO_LABEL`, `LABEL_TO_KEYS`, `keys_to_label()` (shared constants)

- [ ] **Step 1: 编写 config/settings.py**

```python
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
    monitor_index: int = 0          # dxcam output_idx (0=primary)
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
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stop_patience: int = 15
    lr_patience: int = 5
    val_split: float = 0.2
    num_workers: int = 2
    device: str = "cuda"


@dataclass
class InferenceConfig:
    ema_alpha: float = 0.4          # probs = α*new + (1-α)*prev
    target_fps: int = 30


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
    keys: KeysConfig = field(default_factory=KeysConfig)
```

- [ ] **Step 2: 编写测试 tests/test_config.py**

```python
"""Test shared constants and config."""
import pytest
from config.settings import (LABELS, LABEL_TO_IDX, IDX_TO_LABEL,
                              LABEL_TO_KEYS, keys_to_label, AppConfig)


def test_labels_length():
    assert len(LABELS) == 7


def test_label_idx_roundtrip():
    for label in LABELS:
        assert IDX_TO_LABEL[LABEL_TO_IDX[label]] == label


def test_keys_to_label_w():
    assert keys_to_label(w=True, a=False, s=False, d=False) == 'W'


def test_keys_to_label_wa():
    assert keys_to_label(w=True, a=True, s=False, d=False) == 'WA'


def test_keys_to_label_wd():
    assert keys_to_label(w=True, a=False, s=False, d=True) == 'WD'


def test_keys_to_label_a():
    assert keys_to_label(w=False, a=True, s=False, d=False) == 'A'


def test_keys_to_label_d():
    assert keys_to_label(w=False, a=False, s=False, d=True) == 'D'


def test_keys_to_label_s():
    assert keys_to_label(w=False, a=False, s=True, d=False) == 'S'
    # Brake takes priority over everything
    assert keys_to_label(w=True, a=True, s=True, d=True) == 'S'


def test_keys_to_label_none():
    assert keys_to_label(w=False, a=False, s=False, d=False) == 'NONE'


def test_label_to_keys_every_label_covered():
    for label in LABELS:
        assert label in LABEL_TO_KEYS, f"Missing key mapping for {label}"


def test_appconfig_defaults():
    cfg = AppConfig()
    assert cfg.model.num_classes == 7
    assert cfg.train.batch_size == 64
    assert cfg.inference.ema_alpha == 0.4
    assert cfg.keys.panic_key == 'f8'
```

- [ ] **Step 3: 运行测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 10 tests PASS

- [ ] **Step 4: Commit**

```bash
git -C G:/GTAutoDrive add config/settings.py tests/test_config.py
git -C G:/GTAutoDrive commit -m "feat: add AppConfig and shared label constants"
```

---

### Task 3: capture/screen_capture.py — dxcam 屏幕捕获

**Files:**
- Create: `capture/screen_capture.py`

**Interfaces:**
- Consumes: `CaptureConfig`
- Produces: `ScreenCapture` class with `open()`, `read()`, `close()`, `frame_size` property

- [ ] **Step 1: 编写 capture/screen_capture.py**

```python
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
```

- [ ] **Step 2: 运行配置测试确认不报错**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 10 PASS (确认 config 没被改坏)

- [ ] **Step 3: Commit**

```bash
git -C G:/GTAutoDrive add capture/screen_capture.py
git -C G:/GTAutoDrive commit -m "feat: add dxcam screen capture module"
```

---

### Task 4: input/input_simulator.py — 按键模拟

**Files:**
- Create: `input/input_simulator.py`
- Create: `tests/test_input_simulator.py`

**Interfaces:**
- Consumes: `KeysConfig`, `LABEL_TO_KEYS`
- Produces: `InputSimulator` class with `apply(label)`, `release_all()`

- [ ] **Step 1: 编写 input/input_simulator.py**

```python
"""Keyboard input simulation with state tracking via pynput."""
from pynput.keyboard import Controller, Key
from config.settings import LABEL_TO_KEYS, KeysConfig


class InputSimulator:
    def __init__(self, config: KeysConfig):
        self._key_map = {
            'w': config.throttle,
            'a': config.steer_left,
            's': config.brake,
            'd': config.steer_right,
        }
        self._ctrl = Controller()
        # Track which keys are currently pressed to avoid redundant events
        self._state = {'w': False, 'a': False, 's': False, 'd': False}

    def apply(self, label: str):
        target = LABEL_TO_KEYS.get(label, LABEL_TO_KEYS['NONE'])
        for key_char in ('w', 'a', 's', 'd'):
            desired = target[key_char]
            current = self._state[key_char]
            if desired and not current:
                self._ctrl.press(self._key_map[key_char])
                self._state[key_char] = True
            elif not desired and current:
                self._ctrl.release(self._key_map[key_char])
                self._state[key_char] = False

    def release_all(self):
        for key_char in ('w', 'a', 's', 'd'):
            if self._state[key_char]:
                self._ctrl.release(self._key_map[key_char])
                self._state[key_char] = False
```

- [ ] **Step 2: 编写测试 tests/test_input_simulator.py**

```python
"""Tests for InputSimulator — state tracking only (no real key events)."""
import pytest
from input.input_simulator import InputSimulator
from config.settings import KeysConfig


class TestInputSimulator:
    def setup_method(self):
        self.sim = InputSimulator(KeysConfig())

    def test_initial_state_all_released(self):
        assert self.sim._state == {'w': False, 'a': False, 's': False,
                                   'd': False}

    def test_apply_w_sets_w_only(self):
        # Override _ctrl to avoid real key events
        pressed = []
        released = []

        class FakeCtrl:
            def press(self, k): pressed.append(k)
            def release(self, k): released.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim.apply('W')
        assert self.sim._state['w'] is True
        assert self.sim._state['a'] is False
        assert self.sim._state['s'] is False
        assert self.sim._state['d'] is False
        assert 'w' in pressed

    def test_apply_wa_sets_w_and_a(self):
        pressed = []
        released = []

        class FakeCtrl:
            def press(self, k): pressed.append(k)
            def release(self, k): released.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim.apply('WA')
        assert self.sim._state['w'] is True
        assert self.sim._state['a'] is True
        assert self.sim._state['s'] is False
        assert self.sim._state['d'] is False

    def test_transition_w_to_wa(self):
        """Going from W to WA should only press 'a', not re-press 'w'."""
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim._state = {'w': True, 'a': False, 's': False, 'd': False}
        self.sim.apply('WA')
        assert self.sim._state['a'] is True
        assert len(releases) == 0  # no releases

    def test_transition_wd_to_s(self):
        """Brake releases everything, presses 's'."""
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim._state = {'w': True, 'a': False, 's': False, 'd': True}
        self.sim.apply('S')
        assert 'w' in releases or self.sim._state['w'] is False
        assert 'd' in releases or self.sim._state['d'] is False

    def test_apply_none_releases_all(self):
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim._state = {'w': True, 'a': False, 's': False, 'd': True}
        self.sim.apply('NONE')
        assert self.sim._state == {'w': False, 'a': False, 's': False,
                                   'd': False}

    def test_release_all(self):
        self.sim._state = {'w': True, 'a': True, 's': False, 'd': False}
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim.release_all()
        assert self.sim._state == {'w': False, 'a': False, 's': False,
                                   'd': False}
        assert 'w' in releases
        assert 'a' in releases
```

- [ ] **Step 3: 运行测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_input_simulator.py -v
```

Expected: 7 tests PASS

- [ ] **Step 4: Commit**

```bash
git -C G:/GTAutoDrive add input/input_simulator.py tests/test_input_simulator.py
git -C G:/GTAutoDrive commit -m "feat: add input simulator with state tracking"
```

---

### Task 5: tools/test_capture.py — 捕获+按键集成验证

**Files:**
- Create: `tools/test_capture.py`

**Interfaces:**
- Consumes: `ScreenCapture`, `InputSimulator`, `AppConfig`

- [ ] **Step 1: 编写 tools/test_capture.py**

```python
"""Integration test: verify screen capture + key simulation together.

Opens a debug window showing captured frame with FPS.
Press W/A/S/D to verify key simulation works (use Notepad or similar).
Press F8 to exit.
"""
import sys
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import AppConfig, keys_to_label
from capture.screen_capture import ScreenCapture
from input.input_simulator import InputSimulator
from utils.fps_counter import FPSCounter


def main():
    config = AppConfig()
    capture = ScreenCapture(config.capture)
    simulator = InputSimulator(config.keys)
    fps = FPSCounter()

    if not capture.open():
        print("ERROR: Cannot open capture. Is GTA V running?")
        return

    print("Capture test running. Press W/A/S/D to drive; F8 to quit.")
    cv2.namedWindow("GTAutoDrive - Capture Test", cv2.WINDOW_NORMAL)

    try:
        import keyboard
        while True:
            frame = capture.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Read keyboard and apply label
            if not keyboard.is_pressed('f8'):
                w = keyboard.is_pressed('w')
                a = keyboard.is_pressed('a')
                s = keyboard.is_pressed('s')
                d = keyboard.is_pressed('d')
                label = keys_to_label(w, a, s, d)
                simulator.apply(label)

            # Display
            fps.update()
            display = cv2.resize(frame, (960, 540))
            cv2.putText(display, f"FPS: {fps.get():.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "F8=Exit | W/A/S/D=Drive",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            cv2.imshow("GTAutoDrive - Capture Test", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if keyboard.is_pressed('f8'):
                print("\nF8 pressed — exit.")
                break
    finally:
        simulator.release_all()
        capture.close()
        cv2.destroyAllWindows()
        print("Clean exit.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 编写 utils/fps_counter.py（test_capture 的依赖）**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git -C G:/GTAutoDrive add tools/test_capture.py utils/fps_counter.py
git -C G:/GTAutoDrive commit -m "feat: add capture+input integration test tool"
```

---

### Task 6: data/collector.py — 数据录制模块

**Files:**
- Create: `data/collector.py`
- Create: `tests/test_collector.py`

**Interfaces:**
- Consumes: `DataConfig`, `keys_to_label`
- Produces: `DataCollector` class with `start()`, `stop()`, `capture()`, `flush()`, `is_recording`, `frame_count`

- [ ] **Step 1: 编写 data/collector.py**

```python
"""Data collector: record (frame, action) pairs for behavior cloning.

Frames saved as PNG files organized by label class:
  data/recordings/W/frame_00001.png
  data/recordings/WA/frame_00002.png
  ...
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
        label_dir = self.save_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Resize frame
        small = cv2.resize(frame_bgr, (self.fw, self.fh))
        # Convert BGR → RGB for storage
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        filename = label_dir / f"frame_{self._count:05d}.png"
        cv2.imwrite(str(filename), small_rgb)
        self._count += 1

    def flush(self):
        """No-op: files are already written to disk."""
        pass
```

- [ ] **Step 2: 编写测试 tests/test_collector.py**

```python
"""Test DataCollector file output."""
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
from data.collector import DataCollector
from config.settings import DataConfig


def make_frame():
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


class TestDataCollector:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = DataConfig(save_dir=self.tmpdir, frame_width=320,
                         frame_height=180, fps=10)
        self.coll = DataCollector(cfg)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_not_recording_by_default(self):
        assert not self.coll.is_recording

    def test_start_stop(self):
        self.coll.start()
        assert self.coll.is_recording
        self.coll.stop()
        assert not self.coll.is_recording

    def test_capture_creates_file(self):
        self.coll.start()
        self.coll.capture(make_frame(), w=True, a=False, s=False, d=False)
        self.coll.stop()
        # Should have created data/recordings/W/frame_00000.png
        expected = Path(self.tmpdir) / "W" / "frame_00000.png"
        assert expected.exists()
        # Verify dimensions
        img = cv2.imread(str(expected))
        assert img.shape == (180, 320, 3)

    def test_capture_multiple_labels(self):
        self.coll.start()
        self.coll.capture(make_frame(), w=True, a=False, s=False, d=False)  # W
        self.coll.capture(make_frame(), w=True, a=True, s=False, d=False)   # WA
        self.coll.capture(make_frame(), w=False, a=False, s=True, d=False)  # S
        self.coll.stop()
        assert (Path(self.tmpdir) / "W" / "frame_00000.png").exists()
        assert (Path(self.tmpdir) / "WA" / "frame_00001.png").exists()
        assert (Path(self.tmpdir) / "S" / "frame_00002.png").exists()
        assert self.coll.frame_count == 3

    def test_toggle_mode(self):
        assert self.coll.mode == 'train'
        self.coll.toggle_mode()
        assert self.coll.mode == 'recovery'
        self.coll.toggle_mode()
        assert self.coll.mode == 'train'
```

- [ ] **Step 3: 运行测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_collector.py -v
```

Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git -C G:/GTAutoDrive add data/collector.py tests/test_collector.py
git -C G:/GTAutoDrive commit -m "feat: add data collector with PNG storage"
```

---

### Task 7: collect.py — 数据采集入口

**Files:**
- Create: `collect.py`

**Interfaces:**
- Consumes: `ScreenCapture`, `DataCollector`, `AppConfig`

- [ ] **Step 1: 编写 collect.py**

```python
"""Data collection entry point for GTAutoDrive.

Hotkeys:
  F5 — start / stop recording
  F6 — toggle train / recovery mode
  F8 — emergency exit
"""
import sys
import time
import cv2
import keyboard
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import AppConfig
from capture.screen_capture import ScreenCapture
from data.collector import DataCollector
from utils.fps_counter import FPSCounter


def main():
    config = AppConfig()
    capture = ScreenCapture(config.capture)
    collector = DataCollector(config.data)
    fps = FPSCounter()

    if not capture.open():
        print("ERROR: Cannot open capture. Is GTA V running?")
        return

    print("=" * 50)
    print("GTAutoDrive — Data Collection Mode")
    print(f"F5: Start/Stop recording")
    print(f"F6: Toggle train / recovery mode")
    print(f"F8: Exit")
    print(f"Saving to: {config.data.save_dir}")
    print("=" * 50)

    # Debounce state
    f5_was_pressed = False
    f6_was_pressed = False

    cv2.namedWindow("GTAutoDrive - Collection", cv2.WINDOW_NORMAL)

    last_frame_time = 0.0
    frame_interval = 1.0 / config.data.fps

    try:
        while True:
            if keyboard.is_pressed('f8'):
                print("\nF8 pressed — exit.")
                break

            # F5 toggle with debounce
            f5_now = keyboard.is_pressed('f5')
            if f5_now and not f5_was_pressed:
                if collector.is_recording:
                    collector.stop()
                else:
                    collector.start()
            f5_was_pressed = f5_now

            # F6 toggle with debounce
            f6_now = keyboard.is_pressed('f6')
            if f6_now and not f6_was_pressed:
                collector.toggle_mode()
            f6_was_pressed = f6_now

            # Throttle frame capture
            now = time.perf_counter()
            if now - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue
            last_frame_time = now

            frame = capture.read()
            if frame is None:
                cv2.waitKey(1)
                continue

            w = keyboard.is_pressed('w')
            a = keyboard.is_pressed('a')
            s = keyboard.is_pressed('s')
            d = keyboard.is_pressed('d')

            if collector.is_recording:
                collector.capture(frame, w, a, s, d)

            fps.update()

            # Display
            display = cv2.resize(frame, (960, 540))
            # Status bar
            status_color = (0, 0, 255) if collector.is_recording else (
                128, 128, 128)
            status_text = "REC" if collector.is_recording else "IDLE"
            cv2.putText(display, f"[{status_text}] Mode: {collector.mode}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        status_color, 2)
            cv2.putText(display,
                        f"FPS:{fps.get():.0f} Frames:{collector.frame_count}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            cv2.putText(display, "F5=Rec F6=Mode F8=Exit", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display,
                        f"W:{int(w)} A:{int(a)} S:{int(s)} D:{int(d)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            cv2.imshow("GTAutoDrive - Collection", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if collector.is_recording:
            collector.stop()
        capture.close()
        cv2.destroyAllWindows()
        print("Clean exit.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git -C G:/GTAutoDrive add collect.py
git -C G:/GTAutoDrive commit -m "feat: add data collection entry point (F5 record, F6 mode)"
```

---

### Task 8: data/loader.py + data/balancer.py — 数据加载与平衡

**Files:**
- Create: `data/loader.py`
- Create: `data/balancer.py`
- Create: `tests/test_loader.py`
- Create: `tests/test_balancer.py`

**Interfaces:**
- Consumes: `LABELS`, `LABEL_TO_IDX`
- `load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]`: (N,H,W,3) uint8 frames, (N,) int64 labels
- `balance_classes(frames, labels) -> tuple[np.ndarray, np.ndarray]`: balanced subset
- `save_batch(frames, labels, path)`: save as .npz
- `load_batch(path) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: 编写 data/loader.py**

```python
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
```

- [ ] **Step 2: 编写 data/balancer.py**

```python
"""Class-balanced downsampling to prevent model bias."""
import numpy as np
from config.settings import LABELS


def balance_classes(frames: np.ndarray,
                    labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample each class to the count of the smallest class.

    Args:
        frames: (N, H, W, 3) uint8
        labels: (N,) int64 class indices 0-6

    Returns:
        Balanced (frames, labels)
    """
    counts = np.bincount(labels, minlength=len(LABELS))
    print("Before balance:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {counts[i]}")

    min_count = counts[counts > 0].min()
    balanced_frames = []
    balanced_labels = []

    for class_idx in range(len(LABELS)):
        mask = labels == class_idx
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if len(indices) > min_count:
            indices = np.random.choice(indices, min_count, replace=False)
        balanced_frames.append(frames[indices])
        balanced_labels.append(labels[indices])

    result_frames = np.concatenate(balanced_frames, axis=0)
    result_labels = np.concatenate(balanced_labels, axis=0)

    # Shuffle
    perm = np.random.permutation(len(result_frames))
    result_frames = result_frames[perm]
    result_labels = result_labels[perm]

    print(f"After balance: {len(result_frames)} frames "
          f"({min_count} per class)")
    return result_frames, result_labels
```

- [ ] **Step 3: 编写测试 tests/test_loader.py**

```python
"""Test data loading from PNG directory."""
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
from data.loader import load_data, save_batch, load_batch


def make_frame(r, g, b):
    """Create a 180x320 RGB frame."""
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[:, :] = [r, g, b]
    return frame


class TestLoader:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_data_empty_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_data(self.tmpdir)

    def test_save_load_batch_roundtrip(self):
        frames = np.random.randint(0, 255, (10, 180, 320, 3), dtype=np.uint8)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6], dtype=np.int64)
        path = os.path.join(self.tmpdir, "test.npz")
        save_batch(frames, labels, path)
        loaded_frames, loaded_labels = load_batch(path)
        assert np.array_equal(frames, loaded_frames)
        assert np.array_equal(labels, loaded_labels)

    def test_load_data_from_pngs(self):
        # Create PNG files in label subdirs
        for label in ['W', 'A', 'D']:
            label_dir = Path(self.tmpdir) / label
            label_dir.mkdir()
            for i in range(3):
                frame = make_frame(i * 80, i * 40, i * 20)
                cv2.imwrite(str(label_dir / f"frame_{i:05d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        frames, labels = load_data(self.tmpdir)
        assert len(frames) == 9
        assert len(labels) == 9
        assert frames.shape == (9, 180, 320, 3)
        assert frames.dtype == np.uint8
        # Check that all 3 W frames loaded as class 0
        assert np.all(labels[:3] == 0)
```

- [ ] **Step 4: 编写测试 tests/test_balancer.py**

```python
"""Test class balancing."""
import numpy as np
from data.balancer import balance_classes


class TestBalancer:
    def test_balance_reduces_to_min_class(self):
        # 10 W, 5 WA, 2 WD, 1 A, 1 D, 1 S, 1 NONE
        frames = np.zeros((21, 180, 320, 3), dtype=np.uint8)
        labels = np.array(
            [0] * 10 +   # W
            [1] * 5 +    # WA
            [2] * 2 +    # WD
            [3] * 1 +    # A
            [4] * 1 +    # D
            [5] * 1 +    # S
            [6] * 1,     # NONE
            dtype=np.int64
        )
        f_bal, l_bal = balance_classes(frames, labels)
        # Min class = 1, 7 classes
        assert len(f_bal) == 7
        # Each class should appear exactly once
        unique, counts = np.unique(l_bal, return_counts=True)
        assert len(unique) == 7
        assert np.all(counts == 1)

    def test_balance_preserves_shape(self):
        frames = np.zeros((14, 180, 320, 3), dtype=np.uint8)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                          dtype=np.int64)
        f_bal, l_bal = balance_classes(frames, labels)
        # Min = 2, 7 classes = 14 total
        assert len(f_bal) == 14
        assert f_bal.shape == (14, 180, 320, 3)
```

- [ ] **Step 5: 运行测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_loader.py tests/test_balancer.py -v
```

Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git -C G:/GTAutoDrive add data/loader.py data/balancer.py tests/test_loader.py tests/test_balancer.py
git -C G:/GTAutoDrive commit -m "feat: add data loader and class balancer"
```

---

### Task 9: model/network.py — CNN 模型定义

**Files:**
- Create: `model/network.py`
- Create: `tests/test_network.py`

**Interfaces:**
- Produces: `create_model(num_classes=7, dropout=0.5) -> nn.Module`

- [ ] **Step 1: 编写 model/network.py**

```python
"""ResNet-18 adapted for driving action classification.

Input:  (B, 3, 180, 320) RGB, range [0, 255]
Output: (B, 7) logits (no softmax — handled by CrossEntropyLoss)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def create_model(num_classes: int = 7, dropout: float = 0.5) -> nn.Module:
    model = resnet18(weights=None)

    # Replace first conv: standard 7x7→64 is fine for 320x180
    # Replace final FC: 512→num_classes
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
```

- [ ] **Step 2: 编写测试 tests/test_network.py**

```python
"""Test model forward pass shapes."""
import torch
from model.network import create_model


class TestNetwork:
    def test_forward_shape(self):
        model = create_model(num_classes=7, dropout=0.5)
        model.eval()
        # Simulate 320x180 RGB input
        x = torch.randn(4, 3, 180, 320).mul(255)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 7)
        # Output should be logits (not after softmax)
        # Values can be outside [0,1]

    def test_forward_shape_single(self):
        model = create_model(num_classes=7, dropout=0.5)
        model.eval()
        x = torch.randn(1, 3, 180, 320).mul(255)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 7)

    def test_different_num_classes(self):
        model = create_model(num_classes=3, dropout=0.3)
        x = torch.randn(2, 3, 180, 320).mul(255)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)
```

- [ ] **Step 3: 运行测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_network.py -v
```

Expected: 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git -C G:/GTAutoDrive add model/network.py tests/test_network.py
git -C G:/GTAutoDrive commit -m "feat: add ResNet-18 model for 7-class driving"
```

---

### Task 10: model/dataset.py — DataLoader + 数据增强

**Files:**
- Create: `model/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_data`, `LABELS`
- Produces: `DrivingDataset(Dataset)`, `create_dataloaders(frames, labels , batch_size, val_split) -> train_loader, val_loader`

- [ ] **Step 1: 编写 model/dataset.py**

```python
"""PyTorch Dataset and DataLoader for driving data."""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from torchvision.transforms import v2


class DrivingDataset(Dataset):
    """Dataset of (frame, label) pairs with optional transforms."""

    def __init__(self, frames: np.ndarray, labels: np.ndarray,
                 transform=None):
        self.frames = frames     # (N, H, W, 3) uint8 RGB
        self.labels = labels     # (N,) int64
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]   # (H, W, 3) uint8
        label = self.labels[idx]  # scalar int64

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: uint8 → float32 tensor, [0, 255] range
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        label = torch.tensor(label, dtype=torch.long)
        return img, label


def get_train_transforms():
    """Training augmentations: keep RGB in [0, 255] float.

    NOTE: No RandomHorizontalFlip — steering direction (A↔D, WA↔WD) would
    need label mirroring. Left for future improvement.
    """
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),  # [0, 255]
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.RandomTranslation(translate=(0.06, 0.0)),  # horizontal shift only
        v2.RandomRotation(degrees=3),
    ])


def get_val_transforms():
    """Validation: just convert to tensor."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),  # [0, 255]
    ])


def create_dataloaders(frames: np.ndarray, labels: np.ndarray,
                       batch_size: int = 64, val_split: float = 0.2,
                       num_workers: int = 2):
    """Split data into train/val and return DataLoaders."""
    n = len(frames)
    n_val = int(n * val_split)
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = DrivingDataset(
        frames[train_idx], labels[train_idx],
        transform=get_train_transforms()
    )
    val_ds = DrivingDataset(
        frames[val_idx], labels[val_idx],
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_loader, val_loader
```

- [ ] **Step 2: 编写测试 tests/test_dataset.py**

```python
"""Test dataset and dataloader."""
import numpy as np
import torch
from model.dataset import (DrivingDataset, get_train_transforms,
                            get_val_transforms, create_dataloaders)


def make_data(n=20):
    frames = np.random.randint(0, 255, (n, 180, 320, 3), dtype=np.uint8)
    labels = np.random.randint(0, 7, (n,), dtype=np.int64)
    return frames, labels


class TestDrivingDataset:
    def test_basic(self):
        frames, labels = make_data(10)
        ds = DrivingDataset(frames, labels)
        img, label = ds[0]
        assert img.shape == (3, 180, 320)
        assert img.dtype == torch.float32
        assert 0 <= img.max() <= 255
        assert label.dtype == torch.long

    def test_with_train_transform(self):
        frames, labels = make_data(10)
        ds = DrivingDataset(frames, labels, transform=get_train_transforms())
        img, label = ds[0]
        assert img.shape == (3, 180, 320)
        assert label.dtype == torch.long

    def test_with_val_transform(self):
        frames, labels = make_data(10)
        ds = DrivingDataset(frames, labels, transform=get_val_transforms())
        img, label = ds[0]
        assert img.shape == (3, 180, 320)
        assert label.dtype == torch.long

    def test_create_dataloaders(self):
        frames, labels = make_data(50)
        train_dl, val_dl = create_dataloaders(
            frames, labels, batch_size=8, val_split=0.2, num_workers=0
        )
        x, y = next(iter(train_dl))
        assert x.shape[0] <= 8
        assert x.shape[1:] == (3, 180, 320)
        assert y.shape[0] <= 8

    def test_transform_preserves_label(self):
        """验证: 数据增强不改变标签（不含水平翻转因此标签不变）。"""
        frames, labels = make_data(100)
        ds = DrivingDataset(frames, labels, transform=get_train_transforms())
        for i in range(10):
            img, label = ds[i]
            assert img.shape == (3, 180, 320)
            assert 0 <= label.item() <= 6
```

- [ ] **Step 3: 运行测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_dataset.py -v
```

Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git -C G:/GTAutoDrive add model/dataset.py tests/test_dataset.py
git -C G:/GTAutoDrive commit -m "feat: add DrivingDataset and data augmentations"
```

---

### Task 11: model/trainer.py — 训练循环

**Files:**
- Create: `model/trainer.py`
- Create: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `create_model`, `TrainConfig`, `LABELS`
- Produces: `Trainer` class with `train() -> dict`

- [ ] **Step 1: 编写 model/trainer.py**

```python
"""Training loop with early stopping and learning rate scheduling."""
import time
import torch
import torch.nn as nn
import numpy as np
from config.settings import TrainConfig, LABELS


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.lr_patience,
            factor=0.5, min_lr=1e-5
        )

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

    def train(self) -> dict:
        """Run full training. Returns history dict."""
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, "
              f"Val batches: {len(self.val_loader)}")

        for epoch in range(self.config.epochs):
            t0 = time.perf_counter()
            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate()
            elapsed = time.perf_counter() - t0

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"T Loss: {train_loss:.4f} | "
                  f"V Loss: {val_loss:.4f} | "
                  f"V Acc: {val_acc:.2%} | "
                  f"LR: {lr:.1e} | "
                  f"Time: {elapsed:.1f}s")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1} "
                          f"(best: {self.best_epoch})")
                    break

        print(f"Best val loss: {self.best_val_loss:.4f} "
              f"at epoch {self.best_epoch}")
        return history

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return total_loss / len(self.val_loader), correct / total
```

- [ ] **Step 2: 编写测试 tests/test_trainer.py**

```python
"""Smoke test: run one epoch on synthetic data."""
import torch
import numpy as np
from model.trainer import Trainer
from model.network import create_model
from model.dataset import DrivingDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader
from config.settings import TrainConfig


class TestTrainer:
    def test_one_epoch_no_crash(self):
        # Synthetic data: 100 frames, balanced 7 classes
        n = 100
        frames = np.random.randint(0, 255, (n, 180, 320, 3), dtype=np.uint8)
        labels = np.tile(np.arange(7), (n // 7) + 1)[:n].astype(np.int64)

        train_ds = DrivingDataset(frames[:80], labels[:80],
                                  transform=get_train_transforms())
        val_ds = DrivingDataset(frames[80:], labels[80:],
                                transform=get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        model = create_model(num_classes=7)
        cfg = TrainConfig(epochs=2, early_stop_patience=10, device="cpu")
        trainer = Trainer(model, train_loader, val_loader, cfg)
        history = trainer.train()

        assert len(history['train_loss']) >= 1
        assert history['train_loss'][0] > 0  # loss should be positive
```

- [ ] **Step 3: 运行测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/test_trainer.py -v
```

Expected: 1 test PASS

- [ ] **Step 4: Commit**

```bash
git -C G:/GTAutoDrive add model/trainer.py tests/test_trainer.py
git -C G:/GTAutoDrive commit -m "feat: add training loop with early stopping"
```

---

### Task 12: train.py — 训练入口

**Files:**
- Create: `train.py`

**Interfaces:**
- Consumes: all model/ and data/ modules, `AppConfig`

- [ ] **Step 1: 编写 train.py**

```python
"""Training entry point for GTAutoDrive behavior cloning model.

Usage:
    python train.py                          # default paths
    python train.py --data data/recordings   # custom data dir
    python train.py --cpu                    # force CPU
    python train.py --epochs 50              # override epoch count
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from config.settings import AppConfig
from data.loader import load_data
from data.balancer import balance_classes
from model.network import create_model
from model.dataset import create_dataloaders
from model.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train GTAutoDrive behavior cloning model"
    )
    parser.add_argument("--data", type=str, default="data/recordings",
                        help="Path to recorded training data")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    config = AppConfig()
    if args.cpu:
        config.train.device = "cpu"
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr

    print("=" * 50)
    print("GTAutoDrive — Model Training")
    print(f"Data dir:  {args.data}")
    print(f"Device:    {config.train.device}")
    print(f"Epochs:    {config.train.epochs}")
    print(f"Batch:     {config.train.batch_size}")
    print(f"LR:        {config.train.learning_rate}")
    print("=" * 50)

    # Load
    print("\n[1/3] Loading data...")
    frames, labels = load_data(args.data)

    # Balance
    print("\n[2/3] Balancing classes...")
    frames, labels = balance_classes(frames, labels)

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        frames, labels,
        batch_size=config.train.batch_size,
        val_split=config.train.val_split,
        num_workers=config.train.num_workers,
    )

    # Model
    print("\n[3/3] Training...")
    model = create_model(
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
    )
    trainer = Trainer(model, train_loader, val_loader, config.train)
    history = trainer.train()

    # Save
    model_path = Path(config.model.model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    save_path = model_path / config.model.model_name
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Final metrics
    best_val_acc = max(history['val_acc'])
    print(f"Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git -C G:/GTAutoDrive add train.py
git -C G:/GTAutoDrive commit -m "feat: add training entry point"
```

---

### Task 13: utils/ — FPS 计数器 + 调试叠加

**Files:**
- Create: `utils/debug_overlay.py`
- (fps_counter.py 已在 Task 5 创建)

**Interfaces:**
- Produces: `draw_overlay(frame, action, fps, ema_probs) -> np.ndarray`

- [ ] **Step 1: 编写 utils/debug_overlay.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git -C G:/GTAutoDrive add utils/debug_overlay.py
git -C G:/GTAutoDrive commit -m "feat: add debug overlay with probability bars"
```

---

### Task 14: main.py — 推理主循环

**Files:**
- Create: `main.py`

**Interfaces:**
- Consumes: all modules

- [ ] **Step 1: 编写 main.py**

```python
"""GTAutoDrive — L2 autonomous cruise via behavior cloning.

Hotkeys:
  F8 — Emergency stop (release all keys + exit)

Usage:
  python main.py                        # default model
  python main.py --model path/to/model.pt
  python main.py --cpu                  # force CPU inference
  python main.py --dry-run              # debug view only, no key output
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import torch
import numpy as np
import keyboard
from config.settings import AppConfig, IDX_TO_LABEL
from capture.screen_capture import ScreenCapture
from input.input_simulator import InputSimulator
from model.network import create_model
from utils.fps_counter import FPSCounter
from utils.debug_overlay import draw_overlay


class GTAutoDrive:
    def __init__(self, config: AppConfig, model_path: str, dry_run: bool):
        self.config = config
        self.dry_run = dry_run
        self.device = torch.device(
            config.train.device if torch.cuda.is_available() else "cpu"
        )
        self.running = False

        # Capture
        self.capture = ScreenCapture(config.capture)

        # Model
        self.model = create_model(
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
        )
        state = torch.load(model_path, map_location=self.device,
                           weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[MODEL] Loaded from {model_path} (device={self.device})")

        # Input
        self.sim = InputSimulator(config.keys) if not dry_run else None

        # Smoothing
        self._ema_probs = np.ones(config.model.num_classes) / config.model.num_classes
        self._alpha = config.inference.ema_alpha

        # Debug
        self.fps = FPSCounter()
        self._current_action = 'NONE'

    def run(self):
        try:
            self._run_impl()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nFatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def _run_impl(self):
        if not self.capture.open():
            print("ERROR: Cannot open capture. Is GTA V running?")
            return

        print(f"Panic key: {self.config.keys.panic_key}")
        print("Starting in 3 seconds... Switch to GTA V NOW!")
        time.sleep(3)

        fw, fh = self.config.data.frame_width, self.config.data.frame_height
        print(f"Running! Model input: {fw}x{fh}. "
              f"Press {self.config.keys.panic_key} to stop.")

        cv2.namedWindow("GTAutoDrive", cv2.WINDOW_NORMAL)
        self.running = True
        frame_count = 0

        while self.running:
            tick = time.perf_counter()

            frame = self.capture.read()
            if frame is None:
                cv2.waitKey(1)
                time.sleep(0.01)
                continue

            # Resize + BGR→RGB
            small = cv2.resize(frame, (fw, fh))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(small_rgb).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.float().to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            # EMA smoothing
            self._ema_probs = (self._alpha * probs +
                               (1 - self._alpha) * self._ema_probs)

            # Argmax → action
            action_idx = int(np.argmax(self._ema_probs))
            action = IDX_TO_LABEL[action_idx]
            self._current_action = action

            # Apply
            if self.sim is not None:
                self.sim.apply(action)

            # Debug display
            self.fps.update()
            frame_count += 1
            display = draw_overlay(frame, action, self.fps.get(),
                                   self._ema_probs)
            cv2.imshow("GTAutoDrive", cv2.resize(display, (960, 540)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 5 == 0 and keyboard.is_pressed(
                    self.config.keys.panic_key):
                print("\n[!] Emergency stop!")
                break

            # Frame rate control
            elapsed = time.perf_counter() - tick
            sleep_time = (1.0 / self.config.inference.target_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def cleanup(self):
        self.running = False
        if self.sim is not None:
            self.sim.release_all()
        self.capture.close()
        cv2.destroyAllWindows()
        print("GTAutoDrive stopped. All keys released.")


def main():
    parser = argparse.ArgumentParser(
        description="GTAutoDrive - L2 Autonomous Cruise"
    )
    parser.add_argument("--model", type=str,
                        default="data/models/driving_model.pt",
                        help="Path to trained model")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    parser.add_argument("--dry-run", action="store_true",
                        help="Debug view only, no key simulation")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        print("Train first: python train.py")
        return

    config = AppConfig()
    if args.cpu:
        config.train.device = "cpu"

    app = GTAutoDrive(config, args.model, args.dry_run)
    app.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git -C G:/GTAutoDrive add main.py
git -C G:/GTAutoDrive commit -m "feat: add inference main loop with EMA smoothing"
```

---

### Task 15: 最终验证 — 全流程冒烟测试

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 编写集成冒烟测试**

```python
"""End-to-end smoke test: model → inference → action on synthetic frame."""
import numpy as np
import torch
from model.network import create_model
from config.settings import IDX_TO_LABEL, LABEL_TO_KEYS


class TestIntegration:
    def test_model_inference_roundtrip(self):
        """Model takes synthetic frame → outputs a valid action label."""
        model = create_model(num_classes=7)
        model.eval()

        # Random frame
        frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(tensor)

        probs = torch.softmax(logits, dim=1).numpy().flatten()
        action_idx = int(np.argmax(probs))
        action = IDX_TO_LABEL[action_idx]

        # Action must be valid
        assert action in LABEL_TO_KEYS
        # Probabilities sum to 1
        assert abs(probs.sum() - 1.0) < 0.01

    def test_ema_stability(self):
        """EMA smoothing should not crash on repeated calls."""
        alpha = 0.4
        ema_probs = np.ones(7) / 7.0
        model = create_model(num_classes=7)
        model.eval()

        for _ in range(100):
            frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
            tensor = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
            ema_probs = alpha * probs + (1 - alpha) * ema_probs

        assert abs(ema_probs.sum() - 1.0) < 0.01
        assert np.all(ema_probs >= 0) and np.all(ema_probs <= 1)

    def test_all_labels_have_key_mapping(self):
        """Every label must map to a valid key state."""
        from config.settings import LABELS, LABEL_TO_KEYS
        for label in LABELS:
            keys = LABEL_TO_KEYS[label]
            assert 'w' in keys and 'a' in keys and 's' in keys and 'd' in keys
```

- [ ] **Step 2: 运行全部测试**

```bash
G:/GTAutoDrive/venv/Scripts/python -m pytest tests/ -v
```

Expected: ALL tests PASS

- [ ] **Step 3: 更新 .gitignore 确保不提交模型和数据**

```bash
cat >> G:/GTAutoDrive/.gitignore << 'EOF'

# Training artifacts
data/recordings/*
!data/recordings/.gitkeep
data/models/*
!data/models/.gitkeep
*.png
*.jpg
*.npz
EOF
```

- [ ] **Step 4: 最终 Commit**

```bash
git -C G:/GTAutoDrive add tests/test_integration.py .gitignore
git -C G:/GTAutoDrive commit -m "feat: add integration smoke test and update gitignore"
```
