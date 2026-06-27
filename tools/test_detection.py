"""Test YOLO object detection on a screenshot or live OBS feed.

Usage:
  python tools/test_detection.py                 # live OBS feed
  python tools/test_detection.py screenshot.png  # static image
"""
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.object_detector import ObjectDetector


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test YOLO object detection")
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to a screenshot. Omit to use live OBS feed.")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    args = parser.parse_args()

    detector = ObjectDetector(confidence_threshold=args.conf)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Cannot read {args.image}")
            return
        dets = detector.detect(frame)
        print(f"Found {len(dets)} objects:")
        for d in dets:
            print(f"  {d.class_name:10s} conf={d.confidence:.2f} "
                  f"cx={d.center_x:.2f} dist={d.distance_est:.2f} "
                  f"bbox=({d.bbox[0]},{d.bbox[1]},{d.bbox[2]},{d.bbox[3]})")
        disp = frame.copy()
        h, w = disp.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, f"{d.class_name} {d.confidence:.2f}",
                        (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        disp = cv2.resize(disp, (960, 540))
        cv2.imshow("Detection Test", disp)
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
        print("Live detection running. Press Q to quit.")
        cv2.namedWindow("Detection Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection Test", 960, 540)
        cv2.setWindowProperty("Detection Test", cv2.WND_PROP_TOPMOST, 1)
        while True:
            frame = capture.read()
            if frame is None:
                cv2.waitKey(1)
                continue
            dets = detector.detect(frame)
            disp = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp, f"{d.class_name} {d.confidence:.2f} d={d.distance_est:.2f}",
                            (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.putText(disp, f"Objects: {len(dets)}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            disp = cv2.resize(disp, (960, 540))
            cv2.imshow("Detection Test", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
