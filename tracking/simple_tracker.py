import math
import time


MAX_MATCH_DIST = 80


class SimpleTracker:

    def __init__(self):

        self.tracks = {}
        self.next_id = 0

    def update(self,detections):

        now = time.time()

        for tid in self.tracks:
            self.tracks[tid]["matched"] = False

        for cx,cy,x,y,w,h in detections:

            best_id = None
            best_dist = None

            for tid,data in self.tracks.items():

                px,py = data["pos"]

                d = math.hypot(cx-px,cy-py)

                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_dist is not None and best_dist < MAX_MATCH_DIST:

                data = self.tracks[best_id]

                px,py = data["pos"]

                dt = now - data["time"]

                dist = math.hypot(cx-px,cy-py)

                speed = dist/dt if dt>0 else 0

                data["pos"]=(cx,cy)
                data["time"]=now
                data["bbox"]=(x,y,w,h)
                data["speed"]=speed
                data["matched"]=True

            else:

                tid=self.next_id
                self.next_id+=1

                self.tracks[tid]={
                    "pos":(cx,cy),
                    "time":now,
                    "bbox":(x,y,w,h),
                    "speed":0,
                    "matched":True
                }

        results=[]

        for tid,data in self.tracks.items():

            x,y,w,h=data["bbox"]

            cx,cy=data["pos"]

            speed=data["speed"]

            results.append((tid,cx,cy,x,y,w,h,speed))

        return results
