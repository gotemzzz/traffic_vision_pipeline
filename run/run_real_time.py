import cv2
import time
import threading
from picamera2 import Picamera2
from detector.coral_yolo_detector import CoralYOLODetector
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk
from drawing.overlay import draw_tracks


def run_real_time(args):
    width = args.width
    height = args.height
    stop_line_rel = args.stop_line
    detect_every = args.detect_every
    draw_every = args.draw_every

    # ---------------- CAMERA INIT (main thread) ----------------
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()

    # ---------------- GLOBALS ----------------
    latest_frame = None
    frame_lock = threading.Lock()
    running = True

    # Dynamically select the right parser based on model name
    if "yolo" in args.model.lower():
        from detector.coral_yolo_detector import CoralYOLODetector
        detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
    else:
        from detector.coral_tfod_detector import CoralTFODDetector
        detector = CoralTFODDetector(args.model, conf_threshold=args.conf)
        
    tracker = SimpleTracker()
    red_phase = False

    tepoch = time.time()

    # ---------------- CAMERA THREAD ----------------
    def camera_thread():
        nonlocal latest_frame, running

        while running:
            frame = picam2.capture_array()
            with frame_lock:
                latest_frame = frame
            time.sleep(0.001)

    threading.Thread(target=camera_thread, daemon=True).start()

    # ---------------- MAIN LOOP ----------------
    frame_counter = 0
    tracks = []
    print("q = quit | r = toggle red light")

    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        h, w, _ = frame.shape
        stop_line_y = int(stop_line_rel * h)

        frame_counter += 1

        # ---------------- TPU DETECTION ----------------
        t0 = time.time()
        if frame_counter % detect_every == 0:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
        else:
            tracks = tracker.update([])

        t1 = time.time()

        # ---------------- DRAWING ----------------
        if frame_counter % draw_every == 0:
            draw_tracks(frame, tracks, evaluate_risk, red_phase, stop_line_y)

        t2 = time.time()

        # ---------------- SHOW ----------------
        t3 = time.time()
        cv2.imshow("Red Light Risk Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        t4 = time.time()

        if key == ord('q'):
            running = False
            break
        if key == ord('r'):
            red_phase = not red_phase
            print("Red phase:", red_phase)

        # ---------------- DEBUG ----------------
        print(f"frame {frame_counter} | seconds {t4-tepoch:.3f} | "
              f"fps {frame_counter/(t4-tepoch):.3f} |\n"
              f"detect+track: {t1-t0:.3f}s | draw: {t2-t1:.3f}s | "
              f"imshow: {t4-t3:.3f}s | total: {t4-t0:.3f}s")

    cv2.destroyAllWindows()
    picam2.stop()
