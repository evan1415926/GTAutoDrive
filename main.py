"""GTAutoDrive — L2 autonomous cruise via perception + rule-based planning.

Hotkeys:
  F8 — Emergency stop (release all keys + exit)

Usage:
  python main.py                    # perception + planning (default)
  python main.py --model <path>     # BC model fallback (legacy)
  python main.py --dry-run          # debug view only, no key output
"""
import sys
import time
import ctypes
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from config.settings import AppConfig
from capture.screen_capture import ScreenCapture
from input.input_simulator import InputSimulator
from vision.object_detector import ObjectDetector
from vision.lane_detector import LaneDetector
from control.planner import Planner
from utils.fps_counter import FPSCounter

# ── Win32 key-state helper ─────────────────────────────────────────────
_user32 = ctypes.windll.user32
_VK_MAP = {'f8': 0x77, 'q': 0x51}


def _is_key_down(name: str) -> bool:
    return bool(_user32.GetAsyncKeyState(_VK_MAP.get(name, 0)) & 0x8000)


class GTAutoDrive:
    def __init__(self, config: AppConfig, model_path: str | None, dry_run: bool):
        self.config = config
        self.dry_run = dry_run
        self.running = False

        # Capture
        self.capture = ScreenCapture(config.capture)

        # Perception
        pc = config.perception
        self.detector = ObjectDetector(
            confidence_threshold=pc.detection_conf,
            input_size=pc.detection_size,
        )
        self.lane_detector = LaneDetector(
            roi_top_frac=pc.lane_roi_top,
            n_windows=pc.lane_n_windows,
            window_margin=pc.lane_window_margin,
            min_pixels=pc.lane_min_pixels,
        )

        # Planner
        pl = config.planner
        self.planner = Planner(
            emergency_brake_dist=pl.emergency_brake_dist,
            obstacle_brake_dist=pl.obstacle_brake_dist,
            lane_offset_thresh=pl.lane_offset_thresh,
            steer_gain=pl.steer_gain,
            curve_steer_gain=pl.curve_steer_gain,
        )

        # BC model (optional legacy fallback)
        self._bc_model = None
        self._use_bc = model_path is not None
        if self._use_bc:
            import torch
            from model.network import create_model
            self._device = torch.device(
                config.train.device if torch.cuda.is_available() else "cpu"
            )
            state = torch.load(model_path, map_location=self._device,
                               weights_only=True)
            in_ch = state['conv1.weight'].shape[1]
            self._bc_model = create_model(
                num_classes=config.model.num_classes,
                dropout=config.model.dropout,
                input_channels=in_ch,
            )
            self._bc_model.load_state_dict(state)
            self._bc_model.to(self._device)
            self._bc_model.eval()
            self._ema_probs = np.ones(config.model.num_classes) / config.model.num_classes
            self._alpha = config.inference.ema_alpha
            self._conf = config.inference.confidence_threshold
            self._w_bias = config.inference.w_bias
            print(f"[BC] Legacy model loaded: {model_path}")
            print("[MODE] BC model fallback")

        # Input
        self.sim = InputSimulator(config.keys) if not dry_run else None

        # Debug
        self.fps = FPSCounter()
        self._current_action = 'NONE'
        self._last_reason = ''

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

        mode_str = "BC model" if self._use_bc else "perception+planning"
        print(f"[MODE] {mode_str}")
        print(f"Panic key: {self.config.keys.panic_key}")
        print("Starting in 3 seconds... Switch to GTA V NOW!")
        time.sleep(3)
        print(f"Running! Press {self.config.keys.panic_key} to stop.")

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

            if self._use_bc:
                action = self._bc_infer(frame)
            else:
                action = self._perception_infer(frame)

            self._current_action = action
            if self.sim is not None:
                self.sim.apply(action)

            self.fps.update()
            frame_count += 1

            # Build debug display
            if self._use_bc:
                disp = self._build_bc_display(frame)
            else:
                disp = self._build_perception_display(frame)

            cv2.imshow(win_name, disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 5 == 0 and _is_key_down(
                    self.config.keys.panic_key):
                print("\n[!] Emergency stop!")
                break

            elapsed = time.perf_counter() - tick
            target = 1.0 / self.config.inference.target_fps
            if elapsed < target:
                time.sleep(target - elapsed)

    def _perception_infer(self, frame_bgr: np.ndarray) -> str:
        """Run perception + planning pipeline on a BGR frame."""
        detections = self.detector.detect(frame_bgr)
        lane = self.lane_detector.process(frame_bgr)
        action = self.planner.plan(lane, detections)
        self._last_reason = action.reason
        # Store detections and lane for display
        self._last_dets = detections
        self._last_lane = lane
        return self._action_to_label(action)

    def _bc_infer(self, frame_bgr: np.ndarray) -> str:
        """Legacy BC model inference."""
        import torch
        from collections import deque
        fw, fh = self.config.data.frame_width, self.config.data.frame_height
        small = cv2.resize(frame_bgr, (fw, fh))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        f_tensor = torch.from_numpy(small_rgb).permute(2, 0, 1).float().div_(255.0)
        f_tensor = f_tensor.to(self._device)
        f_tensor[0] = (f_tensor[0] - 0.485) / 0.229
        f_tensor[1] = (f_tensor[1] - 0.456) / 0.224
        f_tensor[2] = (f_tensor[2] - 0.406) / 0.225
        with torch.no_grad():
            logits = self._bc_model(f_tensor.unsqueeze(0))
        logits[:, 0] += self._w_bias
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        self._ema_probs = self._alpha * probs + (1 - self._alpha) * self._ema_probs
        idx = int(np.argmax(self._ema_probs))
        if self._ema_probs[idx] < self._conf:
            return 'NONE'
        from config.settings import IDX_TO_LABEL
        return IDX_TO_LABEL[idx]

    @staticmethod
    def _action_to_label(action) -> str:
        if action.s:
            return 'S'
        if action.w and action.a:
            return 'WA'
        if action.w and action.d:
            return 'WD'
        if action.w:
            return 'W'
        if action.a:
            return 'A'
        if action.d:
            return 'D'
        return 'NONE'

    def _build_bc_display(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Minimal display for BC mode."""
        h, w = 160, 320
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (w, h), (20, 20, 20), -1)
        fps_val = self.fps.get()
        cv2.putText(panel, f"BC: {self._current_action}  FPS:{fps_val:.0f}",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(panel, "F8:Exit", (w - 58, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
        return panel

    def _build_perception_display(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Debug display with lane overlay + detection boxes + status."""
        fps_val = self.fps.get()
        dets = getattr(self, '_last_dets', [])
        lane = getattr(self, '_last_lane', None)

        # Start with lane overlay on full frame
        if lane is not None:
            disp = self.lane_detector.draw_overlay(frame_bgr, lane)
        else:
            disp = frame_bgr.copy()

        # Draw detection boxes
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} {det.confidence:.2f} d={det.distance_est:.2f}"
            cv2.putText(disp, label, (x1, max(y1 - 5, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Status bar
        h, w = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(disp, f"ACT:{self._current_action} FPS:{fps_val:.0f}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        reason = getattr(self, '_last_reason', '')
        cv2.putText(disp, reason, (w // 2, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Resize for display window
        disp = cv2.resize(disp, (960, 540))
        return disp

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
    parser.add_argument("--model", type=str, default=None,
                        help="Path to BC model (legacy fallback)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference (BC mode only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Debug view only, no key simulation")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override confidence threshold (BC mode)")
    parser.add_argument("--w-bias", type=float, default=None,
                        help="Override W logit bias (BC mode)")
    args = parser.parse_args()

    config = AppConfig()
    if args.cpu:
        config.train.device = "cpu"

    app = GTAutoDrive(config, args.model, args.dry_run)
    if args.conf is not None:
        app._conf = args.conf
    if args.w_bias is not None:
        app._w_bias = args.w_bias
    app.run()


if __name__ == "__main__":
    main()
