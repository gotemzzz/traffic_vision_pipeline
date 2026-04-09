import math
import time


MAX_MATCH_DIST = 80
STALE_TIMEOUT = 0.6  # shorter stale timeout to reduce ghost trails

MIN_MOVE_PX = 2
VELOCITY_EMA_ALPHA = 0.25
MAX_REASONABLE_SPEED = 2500.0


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class SimpleTracker:
    """
    Output tuple:
      (tid, cx, cy, x, y, w, h, speed, violation, vx, vy, matched)
    """

    def __init__(self, fixed_dt=None):
        self.tracks = {}
        self.next_id = 0
        self.fixed_dt = fixed_dt

    def _new_track(self, tid, cx, cy, x, y, w, h, now):
        self.tracks[tid] = {
            "pos": (cx, cy),
            "prev_pos": (cx, cy),
            "time": now,
            "bbox": (x, y, w, h),
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "matched": True,
            "violation": False,
            "violation_frame": None,
        }

    def update(self, detections):
        now = time.time()

        for tid in self.tracks:
            self.tracks[tid]["matched"] = False

        claimed = set()

        for cx, cy, x, y, w, h in detections:
            det_bbox = (x, y, w, h)

            best_id = None
            best_score = None  # lower is better

            for tid, data in self.tracks.items():
                if tid in claimed:
                    continue

                px, py = data["pos"]
                d = math.hypot(cx - px, cy - py)

                if d > MAX_MATCH_DIST:
                    continue

                iou = _iou_xywh(det_bbox, data["bbox"])

                # association score: prioritize distance, reward IoU overlap
                score = d - (25.0 * iou)

                if best_score is None or score < best_score:
                    best_score = score
                    best_id = tid

            if best_id is not None:
                data = self.tracks[best_id]
                claimed.add(best_id)

                px, py = data["pos"]
                prev_vx = data["vx"]
                prev_vy = data["vy"]

                dt = self.fixed_dt if self.fixed_dt is not None else (now - data["time"])
                if dt <= 1e-6:
                    dt = 1e-3

                dx = cx - px
                dy = cy - py
                dist = math.hypot(dx, dy)

                if dist < MIN_MOVE_PX:
                    inst_vx = 0.0
                    inst_vy = 0.0
                else:
                    inst_vx = dx / dt
                    inst_vy = dy / dt

                inst_speed = math.hypot(inst_vx, inst_vy)
                if inst_speed > MAX_REASONABLE_SPEED:
                    scale = MAX_REASONABLE_SPEED / inst_speed
                    inst_vx *= scale
                    inst_vy *= scale

                vx = VELOCITY_EMA_ALPHA * inst_vx + (1.0 - VELOCITY_EMA_ALPHA) * prev_vx
                vy = VELOCITY_EMA_ALPHA * inst_vy + (1.0 - VELOCITY_EMA_ALPHA) * prev_vy
                speed = math.hypot(vx, vy)

                data["prev_pos"] = data["pos"]
                data["pos"] = (cx, cy)
                data["time"] = now
                data["bbox"] = det_bbox
                data["vx"] = vx
                data["vy"] = vy
                data["speed"] = speed
                data["matched"] = True
            else:
                tid = self.next_id
                self.next_id += 1
                self._new_track(tid, cx, cy, x, y, w, h, now)

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
            vx = data.get("vx", 0.0)
            vy = data.get("vy", 0.0)
            matched = data.get("matched", False)
            results.append((tid, cx, cy, x, y, w, h, speed, violation, vx, vy, matched))

        return results
