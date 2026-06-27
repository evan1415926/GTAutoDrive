"""Lane line detection via HSV thresholding + perspective transform + sliding window.

Pipeline:
  1. Crop ROI (bottom portion of frame — where road lanes are visible)
  2. HSV threshold for white + yellow lane markings
  3. Perspective transform to bird's-eye view
  4. Sliding-window histogram to locate left/right lane pixels
  5. Quadratic polynomial fit (x = a*y² + b*y + c)
  6. Compute vehicle offset from lane center + lane curvature
"""
import cv2
import numpy as np
from models.data_models import LaneInfo


class LaneDetector:
    def __init__(self,
                 roi_top_frac: float = 0.55,       # ROI starts at this fraction of frame height
                 hsv_white_lower: tuple = (0, 0, 180),
                 hsv_white_upper: tuple = (180, 40, 255),
                 hsv_yellow_lower: tuple = (18, 80, 120),
                 hsv_yellow_upper: tuple = (40, 255, 255),
                 persp_top_y: float = 0.40,        # vanishing point Y in ROI (0=top of ROI)
                 persp_margin_top: float = 0.15,    # horizontal margin at top (fraction of frame width)
                 persp_margin_bot: float = 0.10,    # horizontal margin at bottom
                 perspective_src: tuple | None = None,
                 perspective_dst: tuple | None = None,
                 n_windows: int = 9,
                 window_margin: int = 80,
                 min_pixels: int = 50):
        self._roi_top = roi_top_frac
        self._white_lo = np.array(hsv_white_lower)
        self._white_hi = np.array(hsv_white_upper)
        self._yellow_lo = np.array(hsv_yellow_lower)
        self._yellow_hi = np.array(hsv_yellow_upper)
        self._persp_top_y = persp_top_y
        self._persp_margin_top = persp_margin_top
        self._persp_margin_bot = persp_margin_bot
        self._src = perspective_src
        self._dst = perspective_dst
        self._n_windows = n_windows
        self._margin = window_margin
        self._min_pix = min_pixels

        # Cached perspective matrices (recomputed when frame size changes)
        self._M = None
        self._Minv = None
        self._last_shape = None
        self._bird_h = 360
        self._bird_w = 640

    def process(self, frame_bgr: np.ndarray) -> LaneInfo:
        """Detect lanes in a BGR frame.

        Returns LaneInfo with polynomial fits + offset + curvature.
        """
        h, w = frame_bgr.shape[:2]
        self._ensure_perspective(w, h)

        # 1. Crop ROI
        roi = frame_bgr[int(h * self._roi_top):, :]

        # 2. HSV threshold for lane markings
        mask = self._extract_lane_mask(roi)

        # 3. Perspective transform to bird's-eye view
        bird = cv2.warpPerspective(mask, self._M, (self._bird_w, self._bird_h))

        # 4. Find lane pixels via sliding window histogram
        left_fit, right_fit, left_ok, right_ok = self._sliding_window(bird)

        info = LaneInfo(detected=False)

        if left_ok and right_ok and left_fit is not None and right_fit is not None:
            info.left_fit = left_fit
            info.right_fit = right_fit
            info.detected = True

            # 5. Compute metrics at the vehicle position (bottom of bird's-eye)
            y_eval = self._bird_h - 1  # bottom row = closest to vehicle

            left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
            lane_center = (left_x + right_x) / 2.0
            frame_center = self._bird_w / 2.0
            lane_width = abs(right_x - left_x)

            if lane_width > 0:
                info.lane_width = lane_width
                info.center_offset = (frame_center - lane_center) / (lane_width / 2.0)
                # center_offset: <0 = vehicle is left of center (need steer right=D)
                #               >0 = vehicle is right of center (need steer left=A)

            # Curvature (pixel-space, scaled later)
            if left_ok and right_ok:
                l_curv = self._curvature(left_fit, y_eval)
                r_curv = self._curvature(right_fit, y_eval)
                info.curvature = (l_curv + r_curv) / 2.0

            info.confidence = 0.8 if lane_width > 50 else 0.5

        elif left_ok and left_fit is not None:
            info.left_fit = left_fit
            info.detected = True
            info.confidence = 0.3
            # Estimate center from left lane + typical lane width
            y_eval = self._bird_h - 1
            left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
            est_width = 200  # typical lane width in bird's-eye pixels
            lane_center = left_x + est_width / 2
            frame_center = self._bird_w / 2
            info.lane_width = est_width
            info.center_offset = (frame_center - lane_center) / (est_width / 2.0)

        elif right_ok and right_fit is not None:
            info.right_fit = right_fit
            info.detected = True
            info.confidence = 0.3
            y_eval = self._bird_h - 1
            right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
            est_width = 200
            lane_center = right_x - est_width / 2
            frame_center = self._bird_w / 2
            info.lane_width = est_width
            info.center_offset = (frame_center - lane_center) / (est_width / 2.0)

        return info

    # ── internal helpers ────────────────────────────────────────────────

    def _ensure_perspective(self, w, h):
        """Build perspective matrices (lazily, once per frame size)."""
        if self._M is not None and (w, h) == self._last_shape:
            return
        self._last_shape = (w, h)

        if self._src is None:
            # Default source: trapezoid in original image coords (ROI-space)
            roi_h = h - int(h * self._roi_top)
            top_y = int(roi_h * self._persp_top_y)
            bot_y = roi_h - 1
            margin_top = int(w * self._persp_margin_top)
            margin_bot = int(w * self._persp_margin_bot)
            self._src = np.float32([
                (margin_top, top_y),                  # top-left
                (w - margin_top, top_y),              # top-right
                (margin_bot, bot_y),                  # bottom-left
                (w - margin_bot, bot_y),              # bottom-right
            ])

        if self._dst is None:
            self._dst = np.float32([
                (0, 0),
                (self._bird_w, 0),
                (0, self._bird_h),
                (self._bird_w, self._bird_h),
            ])

        self._M = cv2.getPerspectiveTransform(self._src, self._dst)
        self._Minv = cv2.getPerspectiveTransform(self._dst, self._src)

    def _extract_lane_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Combine white and yellow HSV masks for lane markings."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self._white_lo, self._white_hi)
        yellow_mask = cv2.inRange(hsv, self._yellow_lo, self._yellow_hi)
        combined = cv2.bitwise_or(white_mask, yellow_mask)
        # Mild blur to connect dashed lines
        combined = cv2.GaussianBlur(combined, (5, 5), 0)
        return combined

    def _sliding_window(self, bird: np.ndarray):
        """Sliding window search for left and right lane pixels.

        Returns (left_fit, right_fit, left_ok, right_ok).
        Each fit is [a, b, c] for x = a*y² + b*y + c.
        """
        # Histogram of bottom half
        half_h = self._bird_h // 2
        histogram = np.sum(bird[half_h:, :], axis=0)

        midpoint = self._bird_w // 2
        left_base = int(np.argmax(histogram[:midpoint]))
        right_base = int(np.argmax(histogram[midpoint:])) + midpoint

        window_h = self._bird_h // self._n_windows

        nonzero = bird.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_inds = []
        right_inds = []
        left_current = left_base
        right_current = right_base

        for win in range(self._n_windows):
            y_low = self._bird_h - (win + 1) * window_h
            y_high = self._bird_h - win * window_h

            # Left window
            xl_low = left_current - self._margin
            xl_high = left_current + self._margin
            good_left = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                         (nonzerox >= xl_low) & (nonzerox < xl_high)).nonzero()[0]
            if len(good_left) > self._min_pix:
                left_current = int(np.mean(nonzerox[good_left]))
            left_inds.append(good_left)

            # Right window
            xr_low = right_current - self._margin
            xr_high = right_current + self._margin
            good_right = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                          (nonzerox >= xr_low) & (nonzerox < xr_high)).nonzero()[0]
            if len(good_right) > self._min_pix:
                right_current = int(np.mean(nonzerox[good_right]))
            right_inds.append(good_right)

        left_inds = np.concatenate(left_inds) if left_inds else np.array([], dtype=np.intp)
        right_inds = np.concatenate(right_inds) if right_inds else np.array([], dtype=np.intp)

        left_ok = len(left_inds) > self._min_pix
        right_ok = len(right_inds) > self._min_pix

        left_fit = None
        right_fit = None

        if left_ok:
            ly = nonzeroy[left_inds]
            lx = nonzerox[left_inds]
            left_fit = np.polyfit(ly, lx, 2)

        if right_ok:
            ry = nonzeroy[right_inds]
            rx = nonzerox[right_inds]
            right_fit = np.polyfit(ry, rx, 2)

        return left_fit, right_fit, left_ok, right_ok

    @staticmethod
    def _curvature(fit: np.ndarray, y_eval: float) -> float:
        """Compute curvature from quadratic fit at y_eval.

        curvature = (2*A) / (1 + (2*A*y + B)²)^(3/2)
        """
        A, B, _ = fit
        dy = 2 * A * y_eval + B
        d2y = 2 * A
        return d2y / (1 + dy ** 2) ** 1.5

    # ── visualization for debug overlay ─────────────────────────────────

    def draw_overlay(self, frame_bgr: np.ndarray, info: LaneInfo) -> np.ndarray:
        """Draw detected lanes and metrics on a copy of the frame."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        self._ensure_perspective(w, h)
        roi_top = int(h * self._roi_top)

        if info.detected and info.left_fit is not None and info.right_fit is not None:
            # Generate lane curve points in bird's-eye, then inverse-perspective
            ploty = np.linspace(0, self._bird_h - 1, 30)
            left_x = info.left_fit[0] * ploty ** 2 + info.left_fit[1] * ploty + info.left_fit[2]
            right_x = info.right_fit[0] * ploty ** 2 + info.right_fit[1] * ploty + info.right_fit[2]

            pts_left = np.stack([left_x, ploty], axis=1).astype(np.float32).reshape(-1, 1, 2)
            pts_right = np.stack([right_x, ploty], axis=1).astype(np.float32).reshape(-1, 1, 2)

            # Inverse perspective transform
            pts_left_orig = cv2.perspectiveTransform(pts_left, self._Minv)
            pts_right_orig = cv2.perspectiveTransform(pts_right, self._Minv)

            # Draw with ROI offset
            for pts, color in [(pts_left_orig, (255, 0, 0)), (pts_right_orig, (0, 0, 255))]:
                pts_i = pts.astype(np.int32)
                pts_i[:, 0, 1] += roi_top
                cv2.polylines(out, [pts_i], False, color, 3)

            # Fill lane area
            overlay = out.copy()
            pts_combined = np.vstack([
                pts_left_orig.astype(np.int32),
                pts_right_orig.astype(np.int32)[::-1]
            ])
            pts_combined[:, 0, 1] += roi_top
            cv2.fillPoly(overlay, [pts_combined], (0, 80, 0))
            out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)

            # Metrics text
            offset_str = f"offset:{info.center_offset:+.2f}"
            curv_str = f"curv:{info.curvature:+.4f}"
            cv2.putText(out, offset_str, (10, roi_top + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(out, curv_str, (10, roi_top + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif info.detected:
            cv2.putText(out, "LANE: partial", (10, roi_top + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        else:
            cv2.putText(out, "LANE: none", (10, roi_top + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ROI line
        cv2.line(out, (0, roi_top), (w, roi_top), (100, 100, 100), 1)
        return out
