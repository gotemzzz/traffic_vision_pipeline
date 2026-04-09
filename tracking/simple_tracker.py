import math
import time


MAX_MATCH_DIST = 80
STALE_TIMEOUT = 1.0
SPEED_SMOOTHING = 0.25
MIN_MOVE_PX = 3
STATIONARY_SPEED_THRESHOLD = 5.0

# Perspective correction parameters
PERSPECTIVE_SCALE_ENABLED = True
REFERENCE_BBOX_AREA = 6400  # ~80x80 px as reference (adjust if needed)
MIN_SCALE_FACTOR = 1.0  # Don't scale below 1.0 (closest vehicles)
MAX_SCALE_FACTOR = 3.5  # Max correction (distant vehicles). Tune this!


class SimpleTracker:

    def __init__(self, fixed_dt=None):
        self.tracks = {}
        self.next_id = 0
        self.fixed_dt = fixed_dt

    def update(self, detections):

        now = time.time()

        for tid in self.tracks:
            self.tracks[tid]["matched"] = False

        claimed = set()

        for cx, cy, x, y, w, h in detections:

            best_id = None
            best_dist = None

            for tid, data in self.tracks.items():
                if tid in claimed:
                    continue

                px, py = data["pos"]
                d = math.hypot(cx - px, cy - py)

                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_dist is not None and best_dist < MAX_MATCH_DIST:

                data = self.tracks[best_id]
                claimed.add(best_id)

                px, py = data["pos"]

                if self.fixed_dt is not None:
                    dt = self.fixed_dt
                else:
                    dt = now - data["time"]

                dist = math.hypot(cx - px, cy - py)

                if dist < MIN_MOVE_PX:
                    inst_speed = 0.0
                else:
                    inst_speed = dist / dt if dt > 0 else 0.0

                # Apply perspective correction based on bbox size
                if PERSPECTIVE_SCALE_ENABLED:
                    bbox_area = w * h
                    # Larger bbox = closer to camera = smaller scale factor
                    # Smaller bbox = farther from camera = larger scale factor
                    scale_factor = REFERENCE_BBOX_AREA / max(bbox_area, 1)
                    # Clamp to reasonable range
                    scale_factor = max(MIN_SCALE_FACTOR, min(scale_factor, MAX_SCALE_FACTOR))
                    inst_speed *= scale_factor

                prev_speed = data["speed"]
                if prev_speed == 0:
                    smoothed = inst_speed
                else:
                    smoothed = SPEED_SMOOTHING * inst_speed + (1 - SPEED_SMOOTHING) * prev_speed

                # Lock speed at 0 if it's below stationary threshold
                if smoothed < STATIONARY_SPEED_THRESHOLD:
                    smoothed = 0.0

                data["pos"] = (cx, cy)
                data["time"] = now
                data["bbox"] = (x, y, w, h)
                data["speed"] = smoothed
                data["matched"] = True

            else:

                tid = self.next_id
                self.next_id += 1

                self.tracks[tid] = {
                    "pos": (cx, cy),
                    "time": now,
                    "bbox": (x, y, w, h),
                    "speed": 0,
                    "matched": True,
                    "violation": False,
                    "violation_frame": None
                }

        stale_ids = [
            tid for tid, data in self.tracks.items()
            if not data["matched"] and (now - data["time"]) > STALE_TIMEOUT
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        results = []

        for tid, data in self.tracks.items():
            x, y, w, h = data["bbox"]
            cx, cy = data["pos"]
            speed = data["speed"]
            violation = data.get("violation", False)
            results.append((tid, cx, cy, x, y, w, h, speed, violation))

        return results
