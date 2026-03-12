import cv2
import os
import sys
import glob
import argparse
from detector.coral_yolo_detector import CoralYOLODetector
from tracking.simple_tracker import SimpleTracker
from risk.risk_logic import evaluate_risk

# ---------------- ARGS ----------------
parser = argparse.ArgumentParser(description="Run vehicle detection on a folder of images")
parser.add_argument("--input", "-i", required=True, help="Path to folder of input images")
parser.add_argument("--output", "-o", required=True, help="Path to folder for output images")
parser.add_argument("--model", "-m", default="models/yolov8n_full_integer_quant_edgetpu_192.tflite",
                    help="Path to Edge TPU TFLite model")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
parser.add_argument("--red", action="store_true", help="Simulate red-light phase for risk evaluation")
parser.add_argument("--stop-line", type=float, default=0.7,
                    help="Stop line position as fraction of frame height (default: 0.7)")
parser.add_argument("--no-track", action="store_true",
                    help="Disable tracking (treat each image independently)")
args = parser.parse_args()

# ---------------- VALIDATE ----------------
if not os.path.isdir(args.input):
    print(f"Error: input directory '{args.input}' does not exist")
    sys.exit(1)

os.makedirs(args.output, exist_ok=True)

# Gather image files
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
image_paths = sorted([
    p for p in glob.glob(os.path.join(args.input, "*"))
    if os.path.splitext(p)[1].lower() in IMAGE_EXTS
])

if not image_paths:
    print(f"Error: no images found in '{args.input}'")
    sys.exit(1)

print(f"Found {len(image_paths)} images in '{args.input}'")
print(f"Output will be saved to '{args.output}'")
print(f"Red phase: {'ON' if args.red else 'OFF'} | Stop line: {args.stop_line}")

# ---------------- INIT ----------------
detector = CoralYOLODetector(args.model, conf_threshold=args.conf)
tracker = SimpleTracker()

# ---------------- PROCESS ----------------
for idx, img_path in enumerate(image_paths):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  [{idx+1}/{len(image_paths)}] SKIP (unreadable): {img_path}")
        continue

    h, w, _ = frame.shape
    stop_line_y = int(args.stop_line * h)

    # Detect
    detections = detector.detect(frame)

    # Track (or skip tracking for independent images)
    if args.no_track:
        # Build fake track tuples without persistent IDs
        tracks = []
        for i, det in enumerate(detections):
            cx, cy, x, y, bw, bh = det
            tracks.append((i, cx, cy, x, y, bw, bh, 0))
    else:
        tracks = tracker.update(detections)

    # Draw
    for track in tracks:
        tid, cx, cy, x, y, w_box, h_box, speed = track
        risk = evaluate_risk(args.red, cy, stop_line_y, speed)
        color = (0, 0, 255) if risk else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
        label = f"ID {tid} {int(speed)}px/s" + (" RISK" if risk else "")
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw stop line and phase
    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 255, 255), 2)
    phase = "RED" if args.red else "GREEN"
    cv2.putText(frame, phase, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Save
    filename = os.path.basename(img_path)
    out_path = os.path.join(args.output, filename)
    cv2.imwrite(out_path, frame)

    n_det = len(tracks)
    print(f"  [{idx+1}/{len(image_paths)}] {filename} — {n_det} detection{'s' if n_det != 1 else ''} → {out_path}")

print(f"\nDone! {len(image_paths)} images processed → '{args.output}'")
