# TEAM 17 RED LIGHT RUNNER DETECTION

## SS26 ECE480 Captsone Project ~~ Traffic Infringement Unit

This repository showcases our addition to the project that was set up for us by the previous two groups. The project itself is a real-time red light violation detection unit. It consists of four parts: the state detection unit, the traffic infringement unit, the alarm unit, and the power supply.

This repository specifically focuses on the traffic inringement unit or the __TFU__. The TFU provides the system with real-time vehicle detection and red-light risk assessment running on a Raspberry Pi 4 with a Google Coral Edge TPU. The pipeline captures live camera frames, runs YOLOv8n inference on the TPU, tracks vehicles across frames, and flags vehicles at risk of running a red light.

---

## Pipeline Overview

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌─────────────┐    ┌────────────┐
│  Pi Camera  │ -> │  YOLOv8n INT8    │ -> │ Simple Tracker │ -> │ Risk Logic  │ -> │  Display   │
│ (picamera2) │    │  (Coral Edge TPU)│    │ (ID + speed)   │    │ (red-light) │    │  (OpenCV)  │
└─────────────┘    └──────────────────┘    └────────────────┘    └─────────────┘    └────────────┘
```

1. **Capture** — Live frames from the Pi camera (`real_time` mode) or images from a folder (`images` mode).
2. **Detect** — Each frame is fed to a YOLOv8n model compiled for the Coral Edge TPU. The INT8 quantized output is dequantized, decoded (80 COCO classes), and filtered with Non-Maximum Suppression (NMS). Only vehicle classes (car, motorcycle, bus, truck) are kept.
3. **Track** — A lightweight centroid tracker matches detections across frames by proximity, assigning stable IDs and estimating smoothed pixel-space speed via exponential moving average.
4. **Risk Assessment** — During a red-light phase, vehicles approaching the stop line above a speed threshold are flagged as "at risk" of running the light.
5. **Display** — Bounding boxes with dark label backgrounds, track IDs, speeds, and risk labels are drawn on the frame and shown via OpenCV.

## Project Structure

```
traffic_vision_pipeline/
├── main.py                          # CLI entrypoint — dispatches to real_time, images, or animate
├── run/
│   ├── run_real_time.py             # Live camera detection loop
│   └── run_images.py               # Batch image detection + optional animation playback
├── detector/
│   └── coral_yolo_detector.py       # YOLOv8n Edge TPU inference + NMS + output decoding
├── tracking/
│   └── simple_tracker.py            # Centroid-based multi-object tracker with EMA speed
├── risk/
│   └── risk_logic.py                # Red-light risk evaluation logic
├── drawing/
│   ├── __init__.py
│   └── overlay.py                   # Shared drawing helpers (boxes, labels, stop line)
├── models/
│   └── yolov8n_full_integer_quant_edgetpu_192.tflite  # Quantized YOLOv8n model
├── requirements.txt
├── test_delegate.py                 # Quick sanity check for Edge TPU delegate
└── .gitignore
```

## Hardware Requirements

| Component | Details |
|---|---|
| **Single-board computer** | Raspberry Pi 4 (4 GB+ RAM recommended) |
| **AI accelerator** | [Google Coral USB Accelerator](https://coral.ai/products/accelerator/) |
| **Camera** | Any Pi-compatible CSI camera (tested with IMX296 mono sensor) — only needed for `real_time` mode |
| **OS** | Raspberry Pi OS (64-bit / Bookworm recommended) |
| **Power** | 5V 3A USB-C power supply |

## Software Prerequisites

- Python 3.11+
- `libcamera` and `picamera2` (ships with Raspberry Pi OS) — only needed for `real_time` mode
- Google Coral Edge TPU runtime (`libedgetpu`)
- A display or VNC session (for the OpenCV preview window)

## Setup Guide

### 1. Install the Coral Edge TPU runtime

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```

> **Note:** Use `libedgetpu1-std` for standard clock speed. For maximum performance you can install `libedgetpu1-max` instead, but it draws more power and produces more heat.

### 2. Clone the repository

```bash
git clone https://github.com/gotemzzz/traffic_vision_pipeline.git
cd traffic_vision_pipeline
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Add Your Edge TPU model

Place your quantized TFLite model in the `models/` directory:

```
models/your_full_integer_quant_edgetpu_192.tflite
```

> The yolov8n model originally used is included in the repository, so you are not required to import your own. If you wish, however, you can export one yourself using [Ultralytics](https://docs.ultralytics.com/modes/export/):
> ```bash
> pip install ultralytics
> yolo export model=yolov8n.pt format=edgetpu imgsz=192 # replace with your model
> ```
> Additionally, some prebuilt YOLO models can be found here as well:
> ```
> https://github.com/jveitchmichaelis/edgetpu-yolo/
> or
> https://gweb-coral-full.uc.r.appspot.com/models/object-detection/
> ```
> Make sure that you also replace the line in main.py that loads the model:
> ```
> detector = CoralYOLODetector("models/your_full_integer_quant_edgetpu_192.tflite")
> ```

### 5. Verify the Edge TPU is detected

Plug in the Coral USB Accelerator, then:

```bash
python test_delegate.py
```

You should see:

```
OK
```

If this fails, check that the Coral is plugged into a **USB 3.0 port** (blue) and that `libedgetpu` is installed.

## Usage

### Real-time mode (live camera)

```bash
python main.py real_time
```

With options:

```bash
python main.py real_time --conf 0.20 --stop-line 0.65 --detect-every 2
```

**Controls during real-time mode:**

| Key | Action |
|---|---|
| `r` | Toggle red-light phase (GREEN ↔ RED) |
| `q` | Quit |

### Image batch mode

Run detection on a folder of images:

```bash
python main.py images -i frames/input -o frames/output
```

With red-light simulation and animation playback:

```bash
python main.py images -i frames/input -o frames/output --red --animate --fps 15
```

All image mode options:

```bash
python main.py images \
  -i frames/input \
  -o frames/output \
  --conf 0.20 \
  --red \
  --stop-line 0.65 \
  --no-track \
  --animate \
  --fps 10
```

### Animate mode (play back a folder)

Play an existing folder of images as a video — no detection needed:

```bash
python main.py animate -i frames/output --fps 20
```

**Controls during animation playback:**

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `d` or `→` | Next frame (while paused) |
| `a` or `←` | Previous frame (while paused) |
| `q` or `ESC` | Quit playback |

## Configuration Reference

### Real-time mode (`main.py real_time`)

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `models/yolov8n_...tflite` | Path to Edge TPU TFLite model |
| `--conf` | `0.25` | Confidence threshold |
| `--width` | `640` | Camera capture width |
| `--height` | `480` | Camera capture height |
| `--stop-line` | `0.7` | Stop line position (fraction of frame height) |
| `--detect-every` | `1` | Run detection every N frames |
| `--draw-every` | `1` | Draw overlays every N frames |

### Image mode (`main.py images`)

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | *(required)* | Input image folder |
| `--output`, `-o` | *(required)* | Output image folder |
| `--model`, `-m` | `models/yolov8n_...tflite` | Path to Edge TPU TFLite model |
| `--conf` | `0.25` | Confidence threshold |
| `--red` | off | Simulate red-light phase |
| `--stop-line` | `0.7` | Stop line position (fraction of frame height) |
| `--no-track` | off | Treat each image independently (no tracking) |
| `--animate` | off | Play output images as video after processing |
| `--fps` | `10` | Animation playback speed and assumed frame interval for tracking |

### Animate mode (`main.py animate`)

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | *(required)* | Folder of images to play |
| `--fps` | `10` | Playback speed |

### Module-level parameters

Detection in `detector/coral_yolo_detector.py`:

| Parameter | Default | Description |
|---|---|---|
| `VEHICLE_CLASS_IDS` | `{2, 3, 5, 7}` | COCO class IDs to detect (car, motorcycle, bus, truck) |
| `NMS_IOU_THRESHOLD` | `0.5` | IoU threshold for Non-Maximum Suppression |

Tracking in `tracking/simple_tracker.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_MATCH_DIST` | `80` | Max pixel distance to match detection to existing track |
| `STALE_TIMEOUT` | `1.0` | Seconds before an unmatched track is removed |
| `SPEED_SMOOTHING` | `0.4` | EMA alpha for speed (higher = more responsive) |
| `MIN_MOVE_PX` | `2` | Movements below this are treated as jitter (speed = 0) |

Risk in `risk/risk_logic.py`:

| Parameter | Default | Description |
|---|---|---|
| `MIN_SPEED` | `80` | Minimum speed (px/s) to be considered at risk |
| `MAX_DIST` | `200` | Maximum distance (px) from stop line to trigger risk |

## Troubleshooting

| Issue | Fix |
|---|---|
| `ValueError: Failed to load delegate from libedgetpu.so.1` | Coral USB not plugged in, or `libedgetpu` not installed. Run `test_delegate.py` to verify. |
| `No cameras available` | Check CSI ribbon cable and run `libcamera-hello` to verify the camera works. Only applies to `real_time` mode. |
| Font warnings from Qt | Cosmetic only — does not affect functionality. Install `fonts-dejavu` if it bothers you. |
| Very low FPS | Make sure the Coral is on a USB 3.0 port. Increase `--detect-every` to skip frames. |
| Detections but no risk flags | Risk only triggers during red phase — press `r` in real_time mode or use `--red` in image mode. |
| Overlapping/duplicate boxes | NMS should handle this. Try lowering `NMS_IOU_THRESHOLD` (e.g., `0.3`) for more aggressive suppression. |

## License

This project is provided as-is for educational and prototyping purposes.
