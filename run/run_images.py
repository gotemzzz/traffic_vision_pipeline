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


def run_images(args):
    # ---------------- VALIDATE ----------------
    if not args.input or not args.output:
        print("Error: --input and --output are required for image processing.")
        print("  Use 'python main.py animate -i <folder>' to just play back images.")
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
    print(f"Red phase: {'ON' if args.red else 'OFF'} | Stop line: {args.stop_line}")

    # ---------------- INIT ----------------
    detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
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

        # Draw
        draw_tracks(frame, tracks, evaluate_risk, args.red, stop_line_y)

        # Save — always as PNG for consistency
        filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        out_path = os.path.join(args.output, filename)
        cv2.imwrite(out_path, frame)
        output_paths.append(out_path)

        n_det = len(tracks)
        print(f"  [{idx+1}/{len(image_paths)}] {os.path.basename(img_path)} — "
              f"{n_det} detection{'s' if n_det != 1 else ''} → {out_path}")

    print(f"\nDone! {len(output_paths)} images processed → '{args.output}'")

    # ---------------- ANIMATE ----------------
    if args.animate and output_paths:
        animate(output_paths, args.fps)


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
    delay = max(1, int(1000 / fps))
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

        key = cv2.waitKey(0 if paused else delay) & 0xFF

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
