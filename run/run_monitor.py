import sys
import time
from picamera2 import Picamera2
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk, update_violation_status

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None


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


def run_monitor(args):
    width = args.width
    height = args.height
    stop_line_rel = args.stop_line
    detect_every = args.detect_every

    # Detector selection
    if "yolo" in args.model.lower():
        from detector.coral_yolo_detector import CoralYOLODetector
        detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
    else:
        from detector.coral_tfod_detector import CoralTFODDetector
        detector = CoralTFODDetector(args.model, conf_threshold=args.conf)

    tracker = SimpleTracker()

    # Light sensor REQUIRED in monitor mode
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

    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()

    frame_counter = 0
    t0 = time.time()
    prev_alarm_state = False

    print("[MONITOR] running. Ctrl+C to stop.")
    print(f"[MONITOR] mode: {'DRY-RUN' if dry_run else 'LIVE'} | "
          f"alarm_pin={args.alarm_pin} | sensor=ON (pin {args.light_pin})")

    try:
        while True:
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

            if frame_counter % 10 == 0:
                elapsed = time.time() - t0
                fps = frame_counter / elapsed if elapsed > 0 else 0.0
                print(f"[MONITOR] fps={fps:.1f} red={red_phase} tracks={len(updated_tracks)} "
                      f"viol_now={any_violation_now} alarm={alarm_state}")

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[MONITOR] stopping...")
    finally:
        picam2.stop()
        light_sensor.stop()
        if alarm_enabled and not dry_run:
            GPIO.output(args.alarm_pin, GPIO.LOW)
            GPIO.cleanup(args.alarm_pin)
