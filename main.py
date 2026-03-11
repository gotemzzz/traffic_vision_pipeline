import cv2
import subprocess
import numpy as np
import time
import os

from detector.coral_yolo_detector import CoralYOLODetector
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk


WIDTH = 640
HEIGHT = 480

STOP_LINE_Y_REL = 0.7

SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)

detector = CoralYOLODetector("models/yolov8n_full_integer_quant_edgetpu_192.tflite")
tracker = SimpleTracker()

red_phase = False


cmd = [
    "rpicam-vid",
    "--inline",
    "-t","0",
    "--width",str(WIDTH),
    "--height",str(HEIGHT),
    "--framerate","30",
    "--codec","mjpeg",
    "-o","-",
]

p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)

bytes_buffer = b""

print("q = quit | r = toggle red light")


while True:

    chunk = p.stdout.read(262144)

    if not chunk:
        break

    bytes_buffer += chunk

    a = bytes_buffer.find(b'\xff\xd8')
    b = bytes_buffer.find(b'\xff\xd9')

    if a != -1 and b != -1 and b > a:

        jpg = bytes_buffer[a:b+2]
        bytes_buffer = bytes_buffer[b+2:]

        frame = cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv2.IMREAD_COLOR)

        if frame is None:
            continue

        h,w,_ = frame.shape

        stop_line_y = int(STOP_LINE_Y_REL * h)

        detections = detector.detect(frame)

        tracks = tracker.update(detections)

        for track in tracks:

            tid,cx,cy,x,y,w_box,h_box,speed = track

            risk = evaluate_risk(red_phase,cy,stop_line_y,speed)

            color = (0,255,0)

            if risk:
                color = (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w_box,y+h_box),color,2)

            label = f"ID {tid} {int(speed)}px/s"

            if risk:
                label += " RISK"

            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        cv2.line(frame,(0,stop_line_y),(WIDTH,stop_line_y),(0,255,255),2)

        phase = "RED" if red_phase else "GREEN"

        cv2.putText(frame,phase,(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

        cv2.imshow("Red Light Risk Detection",frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):
            red_phase = not red_phase
            print("Red phase:",red_phase)

p.terminate()
cv2.destroyAllWindows()
