"""Object detection via YOLOv8n for road-relevant objects.

Uses COCO-pretrained YOLOv8n. Only keeps classes relevant to driving:
car, truck, bus, motorcycle, bicycle, person.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from models.data_models import Detection

# COCO class IDs we care about
ROAD_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}


class ObjectDetector:
    def __init__(self, confidence_threshold: float = 0.35,
                 input_size: tuple[int, int] = (640, 384)):
        self._conf = confidence_threshold
        self._input_size = input_size
        self._model = YOLO('yolov8n.pt')
        # Warm up
        dummy = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        self._model(dummy, verbose=False)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR image

        Returns:
            List of Detection objects sorted by distance (nearest first).
        """
        h, w = frame_bgr.shape[:2]

        # Resize for speed
        small = cv2.resize(frame_bgr, self._input_size)
        results = self._model(small, conf=self._conf, verbose=False)

        dets = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in ROAD_CLASSES:
                    continue
                conf = boxes.conf[i].item()
                xyxy = boxes.xyxy[i].cpu().numpy()

                # Scale bbox back to original frame coords
                sx, sy = w / self._input_size[0], h / self._input_size[1]
                x1, y1, x2, y2 = xyxy
                x1, x2 = x1 * sx, x2 * sx
                y1, y2 = y1 * sy, y2 * sy

                center_x = ((x1 + x2) / 2) / w
                # Distance estimate: bbox bottom y relative to frame bottom
                distance_est = 1.0 - min(y2 / h, 1.0)

                dets.append(Detection(
                    class_name=ROAD_CLASSES[cls_id],
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    center_x=center_x,
                    distance_est=distance_est,
                ))

        # Sort nearest first
        dets.sort(key=lambda d: d.distance_est)
        return dets

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size
