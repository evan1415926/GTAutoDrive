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
import ctypes
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import torch
import numpy as np
from config.settings import AppConfig, IDX_TO_LABEL
from capture.screen_capture import ScreenCapture
from input.input_simulator import InputSimulator
from model.network import create_model
from utils.fps_counter import FPSCounter
from utils.debug_overlay import draw_overlay

# ── Win32 key-state helper ─────────────────────────────────────────────
_user32 = ctypes.windll.user32
_VK_MAP = {'f8': 0x77, 'q': 0x51}


def _is_key_down(name: str) -> bool:
    return bool(_user32.GetAsyncKeyState(_VK_MAP.get(name, 0)) & 0x8000)


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

        win_name = "GTAutoDrive"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 960, 540)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        self.running = True
        frame_count = 0

        while self.running:
            tick = time.perf_counter()

            frame = self.capture.read()
            if frame is None:
                cv2.waitKey(1)
                time.sleep(0.01)
                continue

            # Resize + BGR->RGB + normalize
            small = cv2.resize(frame, (fw, fh))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(small_rgb).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.float().div(255).to(self.device)
            # ImageNet normalization (must match training transforms)
            tensor[:, 0] = (tensor[:, 0] - 0.485) / 0.229
            tensor[:, 1] = (tensor[:, 1] - 0.456) / 0.224
            tensor[:, 2] = (tensor[:, 2] - 0.406) / 0.225

            # Inference
            with torch.no_grad():
                logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            # EMA smoothing
            self._ema_probs = (self._alpha * probs +
                               (1 - self._alpha) * self._ema_probs)

            # Argmax -> action
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
            cv2.imshow(win_name, cv2.resize(display, (960, 540)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 5 == 0 and _is_key_down(
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
