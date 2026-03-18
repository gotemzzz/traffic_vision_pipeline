import cv2
import os
import shutil
import sys
import glob
from detector.coral_yolo_detector import CoralYOLODetector
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk
from drawing.overlay import draw_tracks

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def gather_images(directory):
    """Collect and sort all image files in a directory."""
    return sorted([
        p for p in glob.glob(os.path.join(directory, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    ])


def parse_transitions(transitions_str, total_frames):
    """
    Parse transition string like 'green:0-50,red:51-150,green:151-end' into a list of (phase, ranges).
    Returns a list of tuples: [(phase, start, end), ...]
    Validates for overlaps and out-of-bounds ranges.
    """
    if not transitions_str:
        return None
    
    transitions = []
    for segment in transitions_str.split(","):
        segment = segment.strip()
        if not segment:
            continue
        
        parts = segment.split(":")
        if len(parts) != 2:
            print(f"Error: Invalid transition format '{segment}'. Expected 'phase:start-end'")
            sys.exit(1)
        
        phase = parts[0].strip().lower()
        if phase not in ("red", "green"):
            print(f"Error: Phase must be 'red' or 'green', got '{phase}'")
            sys.exit(1)
        
        range_str = parts[1].strip()
        range_parts = range_str.split("-")
        if len(range_parts) != 2:
            print(f"Error: Invalid range format '{range_str}'. Expected 'start-end'")
            sys.exit(1)
        
        try:
            start_str, end_str = range_parts[0].strip(), range_parts[1].strip()
            start = int(start_str)
            end = int(end_str) if end_str != "end" else total_frames - 1
            
            if start < 0 or end >= total_frames:
                print(f"Error: Frame range {start}-{end} out of bounds (0-{total_frames - 1})")
                sys.exit(1)
            
            if start > end:
                print(f"Error: Invalid range {start}-{end} (start > end)")
                sys.exit(1)
            
            transitions.append((phase, start, end))
        except ValueError:
            print(f"Error: Could not parse frame numbers in '{range_str}'")
            sys.exit(1)
    
    # Check for overlaps
    for i, (phase1, start1, end1) in enumerate(transitions):
        for phase2, start2, end2 in transitions[i+1:]:
            if not (end1 < start2 or end2 < start1):
                print(f"Error: Overlapping ranges detected: {start1}-{end1} and {start2}-{end2}")
                sys.exit(1)
    
    return sorted(transitions, key=lambda x: x[1])


def get_red_phase_for_frame(frame_idx, transitions):
    """Determine if frame should be in red phase based on transitions."""
    if transitions is None:
        return False  # Default to green
    
    for phase, start, end in transitions:
        if start <= frame_idx <= end:
            return phase == "red"
    
    return False  # Default to green if no transition matches


def run_images(args):
    # Handle real-time mode separately
    if args.real_time:
        run_images_real_time(args)
        return
    
    # Standard file-processing mode
    # ---------------- VALIDATE ----------------
    if not args.input or not args.output:
        print("Error: --input and --output are required for image processing.")
        print("  Use 'python main.py images --input <folder> --real-time' to process without saving files.")
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f"Error: input directory '{args.input}' does not exist")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    image_paths = gather_images(args.input)

    if not image_paths:
        print(f"Error: no images found in '{args.input}'")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in '{args.input}'")
    print(f"Output will be saved to '{args.output}'")
    
    # Parse transitions if provided
    transitions = None
    if args.transitions:
        transitions = parse_transitions(args.transitions, len(image_paths))
        print(f"Traffic light transitions: {args.transitions}")
    elif args.red:
        print(f"Red phase: ON | Stop line: {args.stop_line}")
    else:
        print(f"Red phase: OFF | Stop line: {args.stop_line}")

    # ---------------- INIT ----------------
    
    # Dynamically select the right parser based on model name
    if "yolo" in args.model.lower():
        from detector.coral_yolo_detector import CoralYOLODetector
        detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
    else:
        from detector.coral_tfod_detector import CoralTFODDetector
        detector = CoralTFODDetector(args.model, conf_threshold=args.conf)
    
    tracker = SimpleTracker(fixed_dt=1.0 / args.fps)

    # Clear output directory
    try:
        shutil.rmtree(args.output)
        os.makedirs(args.output)
    except FileNotFoundError:
        pass
    except Exception as e:
        print("Error clearing output directory: ", e)

    # ---------------- PROCESS ----------------
    output_paths = []

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [{idx+1}/{len(image_paths)}] SKIP (unreadable): {img_path}")
            continue

        # Ensure 3-channel BGR regardless of source format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h, w, _ = frame.shape
        stop_line_y = int(args.stop_line * h)

        # Detect
        detections = detector.detect(frame)

        # Track
        if args.no_track:
            tracks = []
            for i, det in enumerate(detections):
                cx, cy, x, y, bw, bh = det
                tracks.append((i, cx, cy, x, y, bw, bh, 0))
        else:
            tracks = tracker.update(detections)

        # Determine red phase for this frame
        if transitions:
            is_red_phase = get_red_phase_for_frame(idx, transitions)
        else:
            is_red_phase = args.red

        # Draw
        draw_tracks(frame, tracks, evaluate_risk, is_red_phase, stop_line_y)

        # Save — always as PNG for consistency
        filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        out_path = os.path.join(args.output, filename)
        cv2.imwrite(out_path, frame)
        output_paths.append(out_path)

        n_det = len(tracks)
        phase_str = "RED" if is_red_phase else "GREEN"
        print(f"  [{idx+1}/{len(image_paths)}] {os.path.basename(img_path)} — "
              f"{phase_str} | {n_det} detection{'s' if n_det != 1 else ''} → {out_path}")

    print(f"\nDone! {len(output_paths)} images processed → '{args.output}'")

    # ---------------- ANIMATE ----------------
    if args.animate and output_paths:
        animate(output_paths, args.fps)


def run_images_real_time(args):
    """Process and display frames in real-time (inference → render on-the-fly)."""
    # ---------------- VALIDATE ----------------
    if not args.input:
        print("Error: --input is required")
        sys.exit(1)

    if not os.path.isdir(args.input):
        print(f"Error: input directory '{args.input}' does not exist")
        sys.exit(1)

    image_paths = gather_images(args.input)

    if not image_paths:
        print(f"Error: no images found in '{args.input}'")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in '{args.input}'")
    
    # Parse transitions if provided
    transitions = None
    if args.transitions:
        transitions = parse_transitions(args.transitions, len(image_paths))
        print(f"Traffic light transitions: {args.transitions}")
    elif args.red:
        print(f"Red phase: ON | Stop line: {args.stop_line}")
    else:
        print(f"Red phase: OFF | Stop line: {args.stop_line}")
    
    print(f"Target FPS: {args.fps}")
    print("Press 'q' to stop, SPACE to pause/resume, a/d to step frames")

    # ---------------- INIT ----------------
    
    # Dynamically select the right parser based on model name
    if "yolo" in args.model.lower():
        from detector.coral_yolo_detector import CoralYOLODetector
        detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
    else:
        from detector.coral_tfod_detector import CoralTFODDetector
        detector = CoralTFODDetector(args.model, conf_threshold=args.conf)
    
    tracker = SimpleTracker(fixed_dt=1.0 / args.fps)

    cv2.namedWindow("Traffic Vision — Real-Time", cv2.WINDOW_AUTOSIZE)

    target_frame_time = 1000 / args.fps
    paused = False
    idx = 0
    total = len(image_paths)
    frame_times = []  # Track actual frame times for stats

    # Print header
    print(f"\n{'Frame':<8} {'Phase':<8} {'Dets':<6} {'FPS':<8} {'Time':<8}")
    print("-" * 45)

    while idx < total:
        frame_start = cv2.getTickCount()
        
        img_path = image_paths[idx]
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[{idx+1}/{total}] SKIP (unreadable): {os.path.basename(img_path)}")
            idx += 1
            continue

        # Ensure 3-channel BGR regardless of source format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h, w, _ = frame.shape
        stop_line_y = int(args.stop_line * h)

        # Detect
        detections = detector.detect(frame)

        # Track
        if args.no_track:
            tracks = []
            for i, det in enumerate(detections):
                cx, cy, x, y, bw, bh = det
                tracks.append((i, cx, cy, x, y, bw, bh, 0))
        else:
            tracks = tracker.update(detections)

        # Determine red phase for this frame
        if transitions:
            is_red_phase = get_red_phase_for_frame(idx, transitions)
        else:
            is_red_phase = args.red

        # Draw
        draw_tracks(frame, tracks, evaluate_risk, is_red_phase, stop_line_y)

        # Calculate processing time
        processing_time_ms = (cv2.getTickCount() - frame_start) / cv2.getTickFrequency() * 1000
        
        # Add frame info overlay
        frame_label = f"Frame {idx+1}/{total}"
        cv2.putText(frame, frame_label, (20, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        time_label = f"Process: {processing_time_ms:.1f}ms"
        cv2.putText(frame, time_label, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if paused:
            cv2.putText(frame, "PAUSED", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Traffic Vision — Real-Time", frame)

        # Calculate wait time to maintain target FPS
        wait_time = max(1, int(target_frame_time - processing_time_ms))
        key = cv2.waitKey(0 if paused else wait_time) & 0xFF

        # Handle keyboard input
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') or key == 83:
            idx = min(idx + 1, total - 1)
        elif key == ord('a') or key == 81:
            idx = max(idx - 1, 0)
        else:
            if not paused:
                # Track frame time
                frame_times.append(processing_time_ms)
                
                # Print stats
                n_det = len(tracks)
                phase_str = "RED" if is_red_phase else "GREEN"
                avg_fps = 1000 / (sum(frame_times[-30:]) / len(frame_times[-30:])) if frame_times else 0
                print(f"{idx+1:<8} {phase_str:<8} {n_det:<6} {avg_fps:>6.1f}   {processing_time_ms:>6.1f}ms")
                
                idx += 1

    cv2.destroyAllWindows()
    
    if frame_times:
        avg_time = sum(frame_times) / len(frame_times)
        avg_fps = 1000 / avg_time
        print(f"\n--- Summary ---")
        print(f"Frames processed: {len(frame_times)}/{total}")
        print(f"Average frame time: {avg_time:.1f}ms")
        print(f"Average FPS: {avg_fps:.1f}")
    
    print("Playback finished.")


def run_animate(args):
    """Entrypoint for the standalone 'animate' subcommand."""
    if not os.path.isdir(args.input):
        print(f"Error: directory '{args.input}' does not exist")
        sys.exit(1)

    image_paths = gather_images(args.input)

    if not image_paths:
        print(f"Error: no images found in '{args.input}'")
        sys.exit(1)

    animate(image_paths, args.fps)


def animate(image_paths, fps=10):
    """Play output images as a video slideshow in an OpenCV window."""
    target_frame_time = 1000 / fps
    total = len(image_paths)

    print(f"\nPlaying {total} frames at {fps} FPS "
          f"(press 'q' to quit, SPACE to pause/resume, a/d to step)")

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        print(f"Error: could not read first frame: {image_paths[0]}")
        return

    cv2.namedWindow("Traffic Vision — Animate", cv2.WINDOW_AUTOSIZE)

    paused = False
    idx = 0

    while True:
        frame_start = cv2.getTickCount()
        
        frame = cv2.imread(image_paths[idx], cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  Warning: skipping unreadable frame: {image_paths[idx]}")
            idx = (idx + 1) % total
            continue

        label = f"Frame {idx+1}/{total}"
        cv2.putText(frame, label, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if paused:
            cv2.putText(frame, "PAUSED", (20, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Traffic Vision — Animate", frame)

        # Calculate how much time we spent on processing
        elapsed_ms = (cv2.getTickCount() - frame_start) / cv2.getTickFrequency() * 1000
        # Subtract processing time from target frame time
        wait_time = max(1, int(target_frame_time - elapsed_ms))

        key = cv2.waitKey(0 if paused else wait_time) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') or key == 83:
            idx = (idx + 1) % total
        elif key == ord('a') or key == 81:
            idx = (idx - 1) % total
        else:
            if not paused:
                idx += 1
                if idx >= total:
                    idx = 0

    cv2.destroyAllWindows()
    print("Playback finished.")
