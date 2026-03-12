import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Vision Pipeline — vehicle detection & red-light risk assessment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- real_time ----
    rt = subparsers.add_parser("real_time", help="Run live camera detection on a Raspberry Pi")
    rt.add_argument("--model", "-m",
                    default="models/yolov8n_full_integer_quant_edgetpu_192.tflite",
                    help="Path to Edge TPU TFLite model")
    rt.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold (default: 0.25)")
    rt.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    rt.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    rt.add_argument("--stop-line", type=float, default=0.7,
                    help="Stop line Y position as fraction of frame height (default: 0.7)")
    rt.add_argument("--detect-every", type=int, default=1,
                    help="Run detection every N frames (default: 1)")
    rt.add_argument("--draw-every", type=int, default=1,
                    help="Draw overlays every N frames (default: 1)")

    # ---- images ----
    img = subparsers.add_parser("images", help="Run detection on a folder of images")
    img.add_argument("--input", "-i", default=None,
                     help="Path to folder of input images")
    img.add_argument("--output", "-o", default=None,
                     help="Path to folder for output images")
    img.add_argument("--model", "-m",
                     default="models/yolov8n_full_integer_quant_edgetpu_192.tflite",
                     help="Path to Edge TPU TFLite model")
    img.add_argument("--conf", type=float, default=0.25,
                     help="Confidence threshold (default: 0.25)")
    img.add_argument("--red", action="store_true",
                     help="Simulate red-light phase for risk evaluation")
    img.add_argument("--stop-line", type=float, default=0.7,
                     help="Stop line position as fraction of frame height (default: 0.7)")
    img.add_argument("--no-track", action="store_true",
                     help="Disable tracking (treat each image independently)")
    img.add_argument("--animate", action="store_true",
                     help="After processing, play output images as a video slideshow")
    img.add_argument("--fps", type=int, default=10,
                     help="Playback FPS for --animate (default: 10)")

    # ---- animate ----
    anim = subparsers.add_parser("animate", help="Play a folder of images as a video slideshow")
    anim.add_argument("--input", "-i", required=True,
                      help="Path to folder of images to play")
    anim.add_argument("--fps", type=int, default=10,
                      help="Playback FPS (default: 10)")

    # ---- help fallback ----
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "real_time":
        from run.run_real_time import run_real_time
        run_real_time(args)
    elif args.command == "images":
        from run.run_images import run_images
        run_images(args)
    elif args.command == "animate":
        from run.run_images import run_animate
        run_animate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
