import cv2
import subprocess
import numpy as np
import time
import os
import threading
from detector.coral_yolo_detector import CoralYOLODetector
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 640, 480
STOP_LINE_Y_REL = 0.7
SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)
DRAW_EVERY_N_FRAMES = 2  # Reduce drawing frequency

# ---------------- GLOBALS ----------------
latest_frame = None
frame_lock = threading.Lock()
running = True

detector = CoralYOLODetector("models/yolov8n_full_integer_quant_edgetpu_192.tflite")
tracker = SimpleTracker()
red_phase = False

# ---------------- CAMERA THREAD ----------------
def camera_thread():
    global latest_frame, running
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
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    bytes_buffer = b""

    while running:
        chunk = p.stdout.read(262144)
        if not chunk:
            break
        bytes_buffer += chunk
        a = bytes_buffer.find(b'\xff\xd8')
        b = bytes_buffer.find(b'\xff\xd9')
        if a != -1 and b != -1 and b > a:
            jpg = bytes_buffer[a:b+2]
            bytes_buffer = bytes_buffer[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                with frame_lock:
                    latest_frame = frame
    p.terminate()

# Start camera thread
threading.Thread(target=camera_thread, daemon=True).start()

# ---------------- MAIN LOOP ----------------
frame_counter = 0
print("q = quit | r = toggle red light")

while True:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    h, w, _ = frame.shape
    stop_line_y = int(STOP_LINE_Y_REL * h)

    t0 = time.time()
    # ---------------- TPU DETECTION ----------------
    detections = detector.detect(frame)
    t1 = time.time()

    # ---------------- TRACKING ----------------
    tracks = tracker.update(detections)

    # ---------------- DRAWING ----------------
    frame_counter += 1
    
    # Copy tracks and draw every N frames, but always display the frame
    display_frame = frame.copy()  # start with original frame

    if frame_counter % DRAW_EVERY_N_FRAMES == 0:
        for track in tracks:
            tid, cx, cy, x, y, w_box, h_box, speed = track
            risk = evaluate_risk(red_phase, cy, stop_line_y, speed)
            color = (0,0,255) if risk else (0,255,0)
            cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), color, 2)
            label = f"ID {tid} {int(speed)}px/s" + (" RISK" if risk else "")
            cv2.putText(display_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw stop line and phase
        cv2.line(display_frame, (0, stop_line_y), (WIDTH, stop_line_y), (0,255,255), 2)
        phase = "RED" if red_phase else "GREEN"
        cv2.putText(display_frame, phase, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    t2 = time.time()

    # ---------------- SHOW ----------------
    cv2.imshow("Red Light Risk Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        running = False
        break
    if key == ord('r'):
        red_phase = not red_phase
        print("Red phase:", red_phase)

    # ---------------- DEBUG ----------------
    print(f"detect: {t1-t0:.3f}s | draw: {t2-t1:.3f}s")

cv2.destroyAllWindows()
