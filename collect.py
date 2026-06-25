"""Data collection entry point for GTAutoDrive.

Hotkeys:
  F5 — start / stop recording
  F6 — toggle train / recovery mode
  F8 — emergency exit
"""
import sys
import time
import cv2
import numpy as np
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

    win_name = "GTAutoDrive - Collection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 320, 240)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

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

            # Build compact 320x240 display panel
            panel = _build_panel(
                collector, fps.get(), w, a, s, d,
                thumbnail=cv2.resize(frame, (320, 180)),
            )
            cv2.imshow(win_name, panel)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if collector.is_recording:
            collector.stop()
        capture.close()
        cv2.destroyAllWindows()
        print("Clean exit.")


def _build_panel(collector, fps_val, w, a, s, d, thumbnail):
    """Render a compact 320x240 status panel with thumbnail."""
    h, w_px = 240, 320
    panel = np.zeros((h, w_px, 3), dtype=np.uint8)

    # Thumbnail at top
    panel[0:180, 0:320] = thumbnail

    # Status bar (bottom 60px)
    bar_y = 180
    cv2.rectangle(panel, (0, bar_y), (w_px, h), (30, 30, 30), -1)

    # REC indicator (red dot + text)
    if collector.is_recording:
        cv2.circle(panel, (10, bar_y + 15), 6, (0, 0, 255), -1)
        status_text = f"REC"
        status_color = (0, 0, 255)
    else:
        cv2.circle(panel, (10, bar_y + 15), 6, (80, 80, 80), -1)
        status_text = "IDLE"
        status_color = (128, 128, 128)

    cv2.putText(panel, f"{status_text} | {collector.mode} | {collector.frame_count}fr",
                (22, bar_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                status_color, 1)
    cv2.putText(panel, f"FPS:{fps_val:.0f}",
                (22, bar_y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (180, 180, 180), 1)

    # Key state indicators (right side of bar)
    keys = [('W', w), ('A', a), ('S', s), ('D', d)]
    for i, (name, pressed) in enumerate(keys):
        kx = 160 + i * 42
        col = (0, 255, 0) if pressed else (60, 60, 60)
        cv2.rectangle(panel, (kx, bar_y + 5), (kx + 36, bar_y + 27), col, -1)
        cv2.putText(panel, name, (kx + 10, bar_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Hint line
    cv2.putText(panel, "F5:Rec F6:Mode F8:Exit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (120, 120, 120), 1)

    return panel
