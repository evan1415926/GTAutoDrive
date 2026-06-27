"""Rule-based control planner: scene quantities → WASD actions.

Priority (highest to lowest):
  1. Emergency brake — obstacle too close in lane
  2. Obstacle response — obstacle ahead, slow down
  3. Lane keeping — correct offset from lane center
  4. Curve following — steer into curvature
  5. Cruise — default forward
"""
from models.data_models import Detection, LaneInfo, ControlAction


class Planner:
    def __init__(self,
                 emergency_brake_dist: float = 0.10,
                 obstacle_brake_dist: float = 0.22,
                 lane_offset_thresh: float = 0.25,
                 steer_gain: float = 2.5,
                 curve_steer_gain: float = 0.8,
                 obstacle_lane_center_range: float = 0.25):
        """
        Args:
            emergency_brake_dist:  normalized distance (0=near, 1=far) below which we brake hard
            obstacle_brake_dist:   normalized distance below which we slow down
            lane_offset_thresh:    abs(center_offset) above which we steer to correct
            steer_gain:            how aggressively to steer for offset correction
            curve_steer_gain:      how aggressively to steer into curves
            obstacle_lane_center_range: how far from lane center (in normalized coords)
                                   an obstacle can be to still be considered "in lane"
        """
        self.emergency_dist = emergency_brake_dist
        self.brake_dist = obstacle_brake_dist
        self.offset_thresh = lane_offset_thresh
        self.steer_gain = steer_gain
        self.curve_gain = curve_steer_gain
        self.lane_range = obstacle_lane_center_range

    def plan(self, lane: LaneInfo, detections: list[Detection]) -> ControlAction:
        """Compute control action from perception data.

        Args:
            lane: LaneInfo from lane detector
            detections: list of Detection from object detector

        Returns:
            ControlAction with WASD flags and reason string.
        """
        # ── 1. Emergency brake ──────────────────────────────────────────
        for det in detections:
            if det.distance_est < self.emergency_dist:
                if self._in_lane(det, lane):
                    return ControlAction(s=True, reason=f'EMERGENCY: {det.class_name} at {det.distance_est:.2f}')

        # ── 2. Obstacle response ────────────────────────────────────────
        lead = self._closest_in_lane(detections, lane)
        if lead is not None and lead.distance_est < self.brake_dist:
            return ControlAction(reason=f'BRAKE: {lead.class_name} at {lead.distance_est:.2f}')

        # ── 3. Lane centering + curve compensation ───────────────────────
        if lane.detected:
            steer = 0.0

            # Offset correction
            if abs(lane.center_offset) > self.offset_thresh:
                steer -= lane.center_offset * self.steer_gain
                # Negative center_offset (vehicle left of center) → positive steer (right/D) is wrong
                # Wait: center_offset < 0 means vehicle is LEFT of lane center.
                # To correct: need to steer RIGHT (D). So steer value should be positive.
                # steer = -center_offset * gain: if offset < 0, -offset > 0 → steer right ✓

            # Curve compensation
            steer += lane.curvature * self.curve_gain
            # curvature > 0 = left turn → steer left (A = negative) ✓
            # Wait, curvature > 0 means the road curves left (fitting x = a*y²+b*y+c, a>0 means left turn).
            # To follow left turn, need A (steer left, negative in our convention).
            # steer += curvature > 0 → steer positive, which would be D (steer right).
            # Hmm, let me reconsider conventions.

            # Let me define: positive steer = press D (turn right), negative steer = press A (turn left)
            # center_offset: <0 = car left of center → need to go right → D → positive steer
            #   steer -= offset: if offset <0, steer = -(-0.3)*2.5 = +0.75 → D ✓
            # curvature: positive curvature = road curves left → need A → negative steer
            #   steer += curvature: if curvature>0, steer increases → that's D (wrong)
            # Actually let me reconsider. curvature = d2y/(1+dy²)^1.5
            # For a left turn, x increases as y increases (in pixel space, y is down).
            # A>0 in x=A*y²+B*y+C means the parabola opens to the right → right turn!
            # Wait no. In our bird's eye view, y goes from 0 (top/far) to H (bottom/near).
            # So y increases as we go toward the vehicle.
            # The polynomial is x = A*y² + B*y + C.
            # As y increases (toward vehicle), if A>0, x increases → lane goes right → right turn.
            # If A<0, x decreases → lane goes left → left turn.
            # But curvature = 2A / (1+(2Ay+B)²)^1.5, so A and curvature have the same sign.
            # curvature > 0 → A > 0 → right turn → need D → positive steer ✓
            # curvature < 0 → A < 0 → left turn → need A → negative steer ✓

            if steer > 0.3:
                return ControlAction(w=True, d=True, reason=f'STEER R: offset={lane.center_offset:+.2f} curve={lane.curvature:+.3f}')
            elif steer < -0.3:
                return ControlAction(w=True, a=True, reason=f'STEER L: offset={lane.center_offset:+.2f} curve={lane.curvature:+.3f}')
            else:
                return ControlAction(w=True, reason=f'LANE OK: offset={lane.center_offset:+.2f}')

        # ── 4. Default cruise ────────────────────────────────────────────
        return ControlAction(w=True, reason='CRUISE (no lane detected)')

    # ── helpers ─────────────────────────────────────────────────────────

    def _in_lane(self, det: Detection, lane: LaneInfo) -> bool:
        """Check if a detection is roughly in the vehicle's lane."""
        if lane.detected and lane.lane_width > 0:
            # Lane center in normalized coords is ~0.5
            expected = 0.5 + lane.center_offset * 0.5  # rough mapping
            return abs(det.center_x - expected) < self.lane_range
        # No lane info: assume objects near center of frame are in lane
        return abs(det.center_x - 0.5) < self.lane_range * 1.5

    def _closest_in_lane(self, detections: list[Detection],
                         lane: LaneInfo) -> Detection | None:
        """Return the nearest detection that is in the vehicle's lane."""
        for det in detections:
            if self._in_lane(det, lane):
                return det
        return None
