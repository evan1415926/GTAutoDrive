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
