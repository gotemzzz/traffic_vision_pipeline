import math
import time


MAX_MATCH_DIST = 80
STALE_TIMEOUT = 1.0
SPEED_SMOOTHING = 0.4
MIN_MOVE_PX = 2


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

                prev_speed = data["speed"]
                if prev_speed == 0:
                    smoothed = inst_speed
                else:
                    smoothed = SPEED_SMOOTHING * inst_speed + (1 - SPEED_SMOOTHING) * prev_speed

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
                    "matched": True
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
            results.append((tid, cx, cy, x, y, w, h, speed))

        return results
