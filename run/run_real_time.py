import cv2
import time
import threading
import os
from picamera2 import Picamera2
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk
from drawing.overlay import draw_tracks


def run_real_time(args):
    width = args.width
    height = args.height
    stop_line_rel = args.stop_line
    detect_every = args.detect_every
    draw_every = args.draw_every

    # Frame saving config
    save_frames = bool(getattr(args, "save_frames", False))
    save_dir = getattr(args, "save_dir", "output/real_time_frames")
    save_every = max(1, int(getattr(args, "save_every", 1)))
    save_prefix = getattr(args, "save_prefix", "frame")
    saved_count = 0

    if save_frames:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[REALTIME] Saving frames to: {save_dir} (every {save_every} frame(s))")

    # ---------------- CAMERA INIT ----------------
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()

    # ---------------- GLOBALS ----------------
    latest_frame = None
    frame_lock = threading.Lock()
    running = True

    # Detector
    if "yolo" in args.model.lower():
        from detector.coral_yolo_detector import CoralYOLODetector
        detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
    else:
        from detector.coral_tfod_detector import CoralTFODDetector
        detector = CoralTFODDetector(args.model, conf_threshold=args.conf)

    tracker = SimpleTracker()
    red_phase_manual = False  # used when sensor is off

    # Optional light sensor
    light_sensor = None
    if args.light_sensor:
        from sensors.light_sensor import LightSensor
        active_high = getattr(args, "light_active_high", False)
        light_sensor = LightSensor(pin=args.light_pin, active_high=active_high)
        light_sensor.start()

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
    print("q = quit | r = toggle red light (manual mode only)")

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()

            h, w, _ = frame.shape
            stop_line_y = int(stop_line_rel * h)
            frame_counter += 1

            # phase source
            if light_sensor is not None:
                active_red_phase = light_sensor.is_red()
                phase_src = "sensor"
            else:
                active_red_phase = red_phase_manual
                phase_src = "manual"

            # detect/track
            t0 = time.time()
            if active_red_phase:
                if frame_counter % detect_every == 0:
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections)
                else:
                    tracks = tracker.update([])
            else:
                tracks = tracker.update([])
            t1 = time.time()

            # draw
            if frame_counter % draw_every == 0:
                draw_tracks(
                    frame,
                    tracks,
                    evaluate_risk,
                    active_red_phase,
                    stop_line_y,
                    approach_vx=args.approach_vx,
                    approach_vy=args.approach_vy,
                )
            t2 = time.time()

            # NEW: save processed frame
            if save_frames and (frame_counter % save_every == 0):
                saved_count += 1
                out_name = f"{save_prefix}_{saved_count:06d}.png"
                out_path = os.path.join(save_dir, out_name)
                cv2.imwrite(out_path, frame)

            # show
            t3 = time.time()
            cv2.imshow("Red Light Risk Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            t4 = time.time()

            if key == ord("q"):
                running = False
                break

            if key == ord("r"):
                if light_sensor is not None:
                    print("Manual toggle disabled while --light-sensor is enabled.")
                else:
                    red_phase_manual = not red_phase_manual
                    print("Red phase (manual):", red_phase_manual)

            print(
                f"frame {frame_counter} | seconds {t4-tepoch:.3f} | "
                f"fps {frame_counter/(t4-tepoch):.3f} | phase={'RED' if active_red_phase else 'GREEN'} ({phase_src}) |\n"
                f"detect+track: {t1-t0:.3f}s | draw: {t2-t1:.3f}s | "
                f"imshow: {t4-t3:.3f}s | total: {t4-t0:.3f}s"
            )

    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        if light_sensor:
            light_sensor.stop()
        if save_frames:
            print(f"[REALTIME] Saved {saved_count} frame(s) to {save_dir}")
