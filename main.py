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
    rt.add_argument("--conf", type=float, default=0.35,
                    help="Confidence threshold (default: 0.35)")
    rt.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    rt.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    rt.add_argument("--stop-line", type=float, default=0.7,
                    help="Stop line Y position as fraction of frame height (default: 0.7)")
    rt.add_argument("--detect-every", type=int, default=1,
                    help="Run detection every N frames (default: 1)")
    rt.add_argument("--draw-every", type=int, default=1,
                    help="Draw overlays every N frames (default: 1)")
    rt.add_argument("--light-sensor", action="store_true",
                    help="Enable photoresistor gateway (GPIO pin set by --light-pin)")
    rt.add_argument("--light-pin", type=int, default=17,
                    help="GPIO pin for light sensor input (default: 17)")
    rt.add_argument("--approach-vx", type=float, default=0.0,
                    help="Approach direction vector X component (default: 0.0)")
    rt.add_argument("--approach-vy", type=float, default=1.0,
                    help="Approach direction vector Y component (default: 1.0)")
    rt.add_argument("--light-active-high", action="store_true",
                    help="Light sensor is active HIGH (default: active LOW)")

    # NEW: optional frame saving for replay with animate
    rt.add_argument("--save-frames", action="store_true",
                    help="Save processed real-time frames to disk")
    rt.add_argument("--save-dir", type=str, default="output/real_time_frames",
                    help="Directory where real-time frames are saved (default: output/real_time_frames)")
    rt.add_argument("--save-every", type=int, default=1,
                    help="Save every Nth frame (default: 1)")
    rt.add_argument("--save-prefix", type=str, default="frame",
                    help="Filename prefix for saved frames (default: frame)")


    # ---- images ----
    img = subparsers.add_parser("images", help="Run detection on a folder of images")
    img.add_argument("--input", "-i", default=None,
                     help="Path to folder of input images")
    img.add_argument("--output", "-o", default=None,
                     help="Path to folder for output images (not required with --real-time)")
    img.add_argument("--model", "-m",
                     default="models/yolov8n_full_integer_quant_edgetpu_192.tflite",
                     help="Path to Edge TPU TFLite model")
    img.add_argument("--conf", type=float, default=0.35,
                     help="Confidence threshold (default: 0.35)")
    img.add_argument("--red", action="store_true",
                     help="Simulate red-light phase for all frames (use --transitions for per-frame control)")
    img.add_argument("--transitions", type=str, default=None,
                     help="Frame-based traffic light transitions (e.g., 'green:0-50,red:51-150,green:151-end')")
    img.add_argument("--stop-line", type=float, default=0.7,
                     help="Stop line position as fraction of frame height (default: 0.7)")
    img.add_argument("--no-track", action="store_true",
                     help="Disable tracking (treat each image independently)")
    img.add_argument("--real-time", action="store_true",
                     help="Process and display frames in real-time (inference → render on-the-fly, no file output)")
    img.add_argument("--animate", action="store_true",
                     help="After processing, play output images as a video slideshow")
    img.add_argument("--fps", type=int, default=10,
                     help="Playback/inference FPS (default: 10)")
    img.add_argument("--light-sensor", action="store_true",
                     help="Enable photoresistor gateway in --real-time mode")
    img.add_argument("--light-pin", type=int, default=17,
                     help="GPIO pin for light sensor input (default: 17)")
    img.add_argument("--approach-vx", type=float, default=0.0,
                     help="Approach direction vector X component (default: 0.0)")
    img.add_argument("--approach-vy", type=float, default=1.0,
                     help="Approach direction vector Y component (default: 1.0)")
    img.add_argument("--light-active-high", action="store_true",
                    help="Light sensor is active HIGH (default: active LOW)")

    # ---- animate ----
    anim = subparsers.add_parser("animate", help="Play a folder of images as a video slideshow")
    anim.add_argument("--input", "-i", required=True,
                      help="Path to folder of images to play")
    anim.add_argument("--fps", type=int, default=10,
                      help="Playback FPS (default: 10)")
    anim.add_argument("--light-sensor", action="store_true",
                      help="Enable photoresistor gateway (GPIO pin set by --light-pin)")
    anim.add_argument("--light-pin", type=int, default=17,
                      help="GPIO pin for light sensor input (default: 17)")
    anim.add_argument("--alarm-pin", type=int, default=None,
                      help="GPIO output pin for alarm signal (HIGH=alarm)")
    anim.add_argument("--approach-vx", type=float, default=0.0,
                      help="Approach direction vector X component (default: 0.0)")
    anim.add_argument("--approach-vy", type=float, default=1.0,
                      help="Approach direction vector Y component (default: 1.0)")
    anim.add_argument("--light-active-high", action="store_true",
                    help="Light sensor is active HIGH (default: active LOW)")
    anim.add_argument("--alarm-active-high", action="store_true",
                     help="Alarm pin is active HIGH (default: active LOW)")

    # ---- monitor ----
    mon = subparsers.add_parser("monitor", help="Headless production mode (no drawing, alarm GPIO, light sensor required)")
    mon.add_argument("--model", "-m",
                     default="models/yolov8n_full_integer_quant_edgetpu_192.tflite",
                     help="Path to Edge TPU TFLite model")
    mon.add_argument("--conf", type=float, default=0.35,
                     help="Confidence threshold (default: 0.35)")
    mon.add_argument("--width", type=int, default=640, help="Camera width")
    mon.add_argument("--height", type=int, default=480, help="Camera height")
    mon.add_argument("--stop-line", type=float, default=0.7,
                     help="Stop line position as fraction of frame height")
    mon.add_argument("--detect-every", type=int, default=1,
                     help="Run detection every N frames")
    mon.add_argument("--light-pin", type=int, default=17,
                     help="GPIO pin for light sensor (required in monitor mode)")
    mon.add_argument("--alarm-pin", type=int, default=None,
                     help="GPIO output pin for alarm signal (HIGH=alarm)")
    mon.add_argument("--dry-run", action="store_true",
                     help="Do not write GPIO alarm output; only log alarm transitions")
    mon.add_argument("--alarm-on-frames", type=int, default=3,
                     help="Consecutive violating frames required to turn alarm ON")
    mon.add_argument("--alarm-off-frames", type=int, default=8,
                     help="Consecutive clear frames required to turn alarm OFF")
    mon.add_argument("--approach-vx", type=float, default=0.0,
                     help="Approach direction vector X component")
    mon.add_argument("--approach-vy", type=float, default=1.0,
                     help="Approach direction vector Y component")
    mon.add_argument("--light-active-high", action="store_true",
                    help="Light sensor is active HIGH (default: active LOW)")
    mon.add_argument("--alarm-active-high", action="store_true",
                     help="Alarm pin is active HIGH (default: active LOW)")

    # NEW: Image feed support
    mon.add_argument("--image-feed", type=str, default=None,
                     help="Path to folder of images to process instead of live camera (for CARLA testing)")
    mon.add_argument("--fps", type=int, default=10,
                     help="FPS for image feed playback (default: 10)")
    mon.add_argument("--render", action="store_true",
                     help="Enable frame rendering for visual feedback")
    mon.add_argument("--render-every", type=int, default=3,
                     help="Render every Nth frame (default: 3, reduces Pi load)")
    mon.add_argument("--loop-feed", action="store_true",
                     help="Loop image feed when end is reached")

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
    elif args.command == "monitor":
        from run.run_monitor import run_monitor
        run_monitor(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
