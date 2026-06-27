"""Test lane detection on a screenshot or live OBS feed.

Usage:
  python tools/test_lane.py                 # live OBS feed
  python tools/test_lane.py screenshot.png  # static image
"""
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.lane_detector import LaneDetector


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test lane detection")
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to a screenshot. Omit to use live OBS feed.")
    args = parser.parse_args()

    detector = LaneDetector()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Cannot read {args.image}")
            return
        info = detector.process(frame)
        print(f"Lane: detected={info.detected} conf={info.confidence:.2f}")
        print(f"  offset={info.center_offset:+.3f}  curvature={info.curvature:+.4f}  width={info.lane_width:.1f}")
        disp = detector.draw_overlay(frame, info)
        disp = cv2.resize(disp, (960, 540))
        cv2.imshow("Lane Test", disp)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        from capture.screen_capture import ScreenCapture
        from config.settings import AppConfig
        config = AppConfig()
        capture = ScreenCapture(config.capture)
        if not capture.open():
            print("ERROR: Cannot open OBS VirtualCam. Is GTA V running?")
            return
        print("Live lane detection running. Press Q to quit.")
        cv2.namedWindow("Lane Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Test", 960, 540)
        cv2.setWindowProperty("Lane Test", cv2.WND_PROP_TOPMOST, 1)
        while True:
            frame = capture.read()
            if frame is None:
                cv2.waitKey(1)
                continue
            info = detector.process(frame)
            disp = detector.draw_overlay(frame, info)
            disp = cv2.resize(disp, (960, 540))
            cv2.imshow("Lane Test", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
