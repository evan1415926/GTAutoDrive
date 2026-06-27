"""Interactive lane detection calibration with real-time trackbar tuning.

Usage:
  python tools/calibrate_lane.py

Trackbars:
  ROI top       — where lane ROI starts (fraction of frame height)
  Persp top Y   — vanishing point Y in ROI (0=top, 1=bottom)
  Persp marginT — horizontal margin at top of trapezoid
  Persp marginB — horizontal margin at bottom of trapezoid

Press S to save current values to config/settings.py.
Press Q to quit.
"""
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from capture.screen_capture import ScreenCapture
from config.settings import AppConfig
from vision.lane_detector import LaneDetector

WIN = "Lane Calibration"


def nothing(_):
    pass


def main():
    config = AppConfig()
    capture = ScreenCapture(config.capture)
    if not capture.open():
        print("ERROR: Cannot open OBS VirtualCam.")
        return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

    # Trackbars (scale x100 for fractional precision)
    cv2.createTrackbar("ROI top x100", WIN,
                       int(config.perception.lane_roi_top * 100), 80, nothing)
    cv2.createTrackbar("Persp topY x100", WIN,
                       int(config.perception.lane_persp_top_y * 100), 70, nothing)
    cv2.createTrackbar("Persp marginT x100", WIN,
                       int(config.perception.lane_persp_margin_top * 100), 30, nothing)
    cv2.createTrackbar("Persp marginB x100", WIN,
                       int(config.perception.lane_persp_margin_bot * 100), 30, nothing)

    print("[Lane Calibration] Adjust trackbars to align lane lines. S=save, Q=quit.")

    while True:
        frame = capture.read()
        if frame is None:
            cv2.waitKey(1)
            continue

        # Read current trackbar values
        roi_top = cv2.getTrackbarPos("ROI top x100", WIN) / 100.0
        persp_y = cv2.getTrackbarPos("Persp topY x100", WIN) / 100.0
        persp_mt = cv2.getTrackbarPos("Persp marginT x100", WIN) / 100.0
        persp_mb = cv2.getTrackbarPos("Persp marginB x100", WIN) / 100.0

        # Rebuild detector with current params
        detector = LaneDetector(
            roi_top_frac=roi_top,
            persp_top_y=persp_y,
            persp_margin_top=persp_mt,
            persp_margin_bot=persp_mb,
            n_windows=config.perception.lane_n_windows,
            window_margin=config.perception.lane_window_margin,
            min_pixels=config.perception.lane_min_pixels,
        )
        info = detector.process(frame)
        disp = detector.draw_overlay(frame, info)

        # Show current params
        cv2.putText(disp, f"ROI:{roi_top:.2f} topY:{persp_y:.2f} mT:{persp_mt:.2f} mB:{persp_mb:.2f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(disp, "S:save  Q:quit", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        disp = cv2.resize(disp, (960, 540))
        cv2.imshow(WIN, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Persist current values to settings.py
            pc = config.perception
            settings_path = Path(__file__).parent.parent / "config" / "settings.py"
            text = settings_path.read_text(encoding='utf-8')
            import re
            text = re.sub(r'lane_roi_top: float = [\d.]+',
                         f'lane_roi_top: float = {roi_top:.2f}', text)
            text = re.sub(r'lane_persp_top_y: float = [\d.]+',
                         f'lane_persp_top_y: float = {persp_y:.2f}', text)
            text = re.sub(r'lane_persp_margin_top: float = [\d.]+',
                         f'lane_persp_margin_top: float = {persp_mt:.2f}', text)
            text = re.sub(r'lane_persp_margin_bot: float = [\d.]+',
                         f'lane_persp_margin_bot: float = {persp_mb:.2f}', text)
            settings_path.write_text(text, encoding='utf-8')
            print(f"[SAVED] roi_top={roi_top:.2f} topY={persp_y:.2f} "
                  f"marginT={persp_mt:.2f} marginB={persp_mb:.2f}")

    capture.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
