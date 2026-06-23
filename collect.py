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
