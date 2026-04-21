import sys
import time
import os
import glob
import cv2
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk, update_violation_status

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None

try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None


class AlarmLatch:
    """
    Hysteresis / debounce for alarm output.

    - Turn ON only after `on_frames` consecutive violating frames.
    - Turn OFF only after `off_frames` consecutive clear frames.
    """
    def __init__(self, on_frames=3, off_frames=8):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self._violating_count = 0
        self._clear_count = 0
        self.state = False

    def update(self, any_violation_now: bool) -> bool:
        if any_violation_now:
            self._violating_count += 1
            self._clear_count = 0
            if not self.state and self._violating_count >= self.on_frames:
                self.state = True
        else:
            self._clear_count += 1
            self._violating_count = 0
            if self.state and self._clear_count >= self.off_frames:
                self.state = False
        return self.state


def _track_with_violation(track, violation_bool):
    if len(track) >= 12:
        tid, cx, cy, x, y, bw, bh, speed, _, vx, vy, matched = track[:12]
        return (tid, cx, cy, x, y, bw, bh, speed, violation_bool, vx, vy, matched)
    elif len(track) >= 11:
        tid, cx, cy, x, y, bw, bh, speed, _, vx, vy = track[:11]
        return (tid, cx, cy, x, y, bw, bh, speed, violation_bool, vx, vy, True)
    else:
        tid, cx, cy, x, y, bw, bh, speed, _ = track[:9]
        return (tid, cx, cy, x, y, bw, bh, speed, violation_bool, 0.0, 0.0, True)


def gather_images(directory):
    """Collect and sort all image files in a directory."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return sorted([
        p for p in glob.glob(os.path.join(directory, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    ])


def run_monitor(args):
    width = args.width
    height = args.height
    stop_line_rel = args.stop_line
    detect_every = args.detect_every
    
    # NEW: Image feed support
    use_image_feed = args.image_feed is not None
    render_enabled = bool(getattr(args, "render", False))
    render_every = max(1, int(getattr(args, "render_every", 3)))
    loop_feed = bool(getattr(args, "loop_feed", False))
    fps = int(getattr(args, "fps", 10))
    frame_delay_ms = 1000 / fps if fps > 0 else 0

    if use_image_feed:
        if not os.path.isdir(args.image_feed):
            print(f"[MONITOR] ERROR: image feed directory '{args.image_feed}' does not exist")
            sys.exit(1)
        image_paths = gather_images(args.image_feed)
        if not image_paths:
            print(f"[MONITOR] ERROR: no images found in '{args.image_feed}'")
            sys.exit(1)
        print(f"[MONITOR] Image feed mode: {len(image_paths)} images from '{args.image_feed}'")
        print(f"[MONITOR] Rendering: {'ON (every {render_every} frame)' if render_enabled else 'OFF'}")

    # Detector selection
    if "yolo" in args.model.lower():
        from detector.coral_yolo_detector import CoralYOLODetector
        detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
    else:
        from detector.coral_tfod_detector import CoralTFODDetector
        detector = CoralTFODDetector(args.model, conf_threshold=args.conf)

    tracker = SimpleTracker()

    # Light sensor REQUIRED in monitor mode
    light_sensor = None
    try:
        from sensors.light_sensor import LightSensor
        light_sensor = LightSensor(pin=args.light_pin)
        light_sensor.start()
    except Exception as e:
        print(f"[MONITOR] ERROR: light sensor init failed on GPIO {args.light_pin}: {e}")
        print("[MONITOR] monitor mode requires a working light sensor.")
        sys.exit(2)

    # Alarm output setup
    alarm_enabled = args.alarm_pin is not None
    dry_run = bool(args.dry_run)

    if dry_run:
        print("[MONITOR] DRY-RUN enabled: no GPIO alarm output will be written.")

    if alarm_enabled and not dry_run:
        if GPIO is None:
            print("[MONITOR] ERROR: RPi.GPIO not available but --alarm-pin was set without --dry-run")
            light_sensor.stop()
            sys.exit(2)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(args.alarm_pin, GPIO.OUT, initial=GPIO.LOW)

    latch = AlarmLatch(on_frames=args.alarm_on_frames, off_frames=args.alarm_off_frames)

    # Camera OR Image feed
    if not use_image_feed:
        if Picamera2 is None:
            print("[MONITOR] ERROR: Picamera2 not available. Use --image-feed for testing without camera.")
            light_sensor.stop()
            sys.exit(2)
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (width, height)})
        picam2.configure(config)
        picam2.start()
    
    # Setup rendering window if enabled
    if render_enabled:
        cv2.namedWindow("Monitor — Visual Feedback", cv2.WINDOW_AUTOSIZE)

    frame_counter = 0
    t0 = time.time()
    prev_alarm_state = False
    image_idx = 0  # For image feed cycling

    print("[MONITOR] running. Ctrl+C to stop.")
    print(f"[MONITOR] mode: {'DRY-RUN' if dry_run else 'LIVE'} | "
          f"alarm_pin={args.alarm_pin} | sensor=ON (pin {args.light_pin})")

    try:
        while True:
            # Get frame from camera or image feed
            if use_image_feed:
                img_path = image_paths[image_idx % len(image_paths)]
                frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"[MONITOR] WARNING: could not read {img_path}, skipping")
                    image_idx += 1
                    continue
                
                # Resize to target dimensions if needed
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                image_idx += 1
                if image_idx >= len(image_paths):
                    if loop_feed:
                        image_idx = 0
                        print(f"[MONITOR] looping image feed from start")
                    else:
                        print(f"[MONITOR] image feed exhausted, stopping")
                        break
                
                # Frame timing
                if frame_delay_ms > 0:
                    time.sleep(frame_delay_ms / 1000.0)
            else:
                frame = picam2.capture_array()

            h, w, _ = frame.shape
            stop_line_y = int(stop_line_rel * h)
            frame_counter += 1

            # Always from sensor in monitor mode
            red_phase = light_sensor.is_red()

            # Detection gate
            if red_phase and (frame_counter % detect_every == 0):
                detections = detector.detect(frame)
                tracks = tracker.update(detections)
            else:
                tracks = tracker.update([])

            any_violation_now = False
            updated_tracks = []

            for tr in tracks:
                v = update_violation_status(
                    tr,
                    stop_line_y,
                    red_phase,
                    approach_vx=args.approach_vx,
                    approach_vy=args.approach_vy,
                )
                tr2 = _track_with_violation(tr, v)
                updated_tracks.append(tr2)

                tid, cx, cy, x, y, bw, bh, speed, violation, vx, vy, matched = tr2
                risk_now = evaluate_risk(
                    red_phase,
                    cy,
                    stop_line_y,
                    speed,
                    violation_history=violation,
                    vx=vx,
                    vy=vy,
                    approach_vx=args.approach_vx,
                    approach_vy=args.approach_vy,
                )
                if risk_now:
                    any_violation_now = True

            alarm_state = latch.update(any_violation_now)

            if alarm_enabled and not dry_run:
                GPIO.output(args.alarm_pin, GPIO.HIGH if alarm_state else GPIO.LOW)

            if alarm_state != prev_alarm_state:
                state_txt = "ON" if alarm_state else "OFF"
                if dry_run:
                    print(f"[MONITOR][DRY] ALARM -> {state_txt}")
                else:
                    print(f"[MONITOR] ALARM -> {state_txt}")
                prev_alarm_state = alarm_state

            # Optional rendering for visual feedback (minimal overhead - just raw frame + light state)
            if render_enabled and (frame_counter % render_every == 0):
                frame_copy = frame.copy()
                
                # Simple light state indicator (top-left corner)
                phase_color = (0, 0, 255) if red_phase else (0, 255, 0)  # BGR: red or green
                phase_text = "RED" if red_phase else "GREEN"
                cv2.circle(frame_copy, (30, 30), 15, phase_color, -1)
                cv2.putText(frame_copy, phase_text, (55, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)
                
                cv2.imshow("Monitor — Visual Feedback", frame_copy)
                # Non-blocking display update
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            if frame_counter % 10 == 0:
                elapsed = time.time() - t0
                fps_actual = frame_counter / elapsed if elapsed > 0 else 0.0
                print(f"[MONITOR] fps={fps_actual:.1f} red={red_phase} tracks={len(updated_tracks)} "
                      f"viol_now={any_violation_now} alarm={alarm_state}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[MONITOR] stopping...")
    finally:
        if not use_image_feed:
            picam2.stop()
        light_sensor.stop()
        if alarm_enabled and not dry_run:
            GPIO.output(args.alarm_pin, GPIO.LOW)
            GPIO.cleanup(args.alarm_pin)
        if render_enabled:
            cv2.destroyAllWindows()
