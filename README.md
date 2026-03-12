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

1. **Capture** — A background thread grabs 640×480 frames from the Pi camera using `picamera2`.
2. **Detect** — Each frame is resized to 192×192 and fed to a YOLOv8n model compiled for the Coral Edge TPU. The INT8 quantized output is dequantized and decoded (80 COCO classes). Only vehicle classes (car, motorcycle, bus, truck) are kept.
3. **Track** — A lightweight centroid tracker matches detections across frames by proximity, assigning stable IDs and estimating pixel-space speed.
4. **Risk Assessment** — During a red-light phase, vehicles approaching the stop line above a speed threshold are flagged as "at risk" of running the light.
5. **Display** — Bounding boxes, track IDs, speeds, and risk labels are drawn on the frame and shown via OpenCV.

## Project Structure

```
traffic_vision_pipeline/
├── main.py                          # Entry point — capture loop, drawing, keyboard controls
├── detector/
│   └── coral_yolo_detector.py       # YOLOv8n Edge TPU inference + output decoding
├── tracking/
│   └── simple_tracker.py            # Centroid-based multi-object tracker
├── risk/
│   └── risk_logic.py                # Red-light risk evaluation logic
├── models/
│   └── yolov8n_full_integer_quant_edgetpu_192.tflite  # Quantized YOLOv8n model
├── requirements.txt
├── test_delegate.py                 # Quick sanity check for Edge TPU delegate
└── .gitignore
```

---

## Hardware Requirements

| Component | Details |
|---|---|
| **Single-board computer** | Raspberry Pi 4 (4 GB+ RAM recommended) |
| **AI accelerator** | [Google Coral USB Accelerator](https://coral.ai/products/accelerator/) |
| **Camera** | Any Pi-compatible CSI camera (tested with IMX296 mono sensor) |
| **OS** | Raspberry Pi OS (64-bit / Bookworm recommended) |
| **Power** | 5V 3A USB-C power supply |

## Software Prerequisites

- Python 3.11+
- `libcamera` and `picamera2` (ships with Raspberry Pi OS)
- Google Coral Edge TPU runtime (`libedgetpu`)
- A display or VNC session (for the OpenCV preview window)

---

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

### 6. Run the pipeline

```bash
python main.py
```

---

## Controls

| Key | Action |
|---|---|
| `r` | Toggle red-light phase (GREEN ↔ RED) |
| `q` | Quit |

## Batch Mode (Image Folder)

You can also run the pipeline on a folder of images instead of a live camera feed:

```bash
python run_images.py --input ./test_frames --output ./results
```

Run `python run_images.py --help` for all options.

## Configuration

Key parameters can be tuned at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `WIDTH, HEIGHT` | `640, 480` | Camera capture resolution |
| `STOP_LINE_Y_REL` | `0.7` | Stop line position as a fraction of frame height (0 = top, 1 = bottom) |
| `DETECT_EVERY_N_FRAMES` | `1` | Run detection every N frames (increase to trade accuracy for speed) |
| `DRAW_EVERY_N_FRAMES` | `1` | Draw overlays every N frames |

Risk thresholds in `risk/risk_logic.py`:

| Parameter | Default | Description |
|---|---|---|
| `MIN_SPEED` | `80` | Minimum speed (px/s) to be considered at risk |
| `MAX_DIST` | `200` | Maximum distance (px) from stop line to trigger risk |

Detection settings in `detector/coral_yolo_detector.py`:

| Parameter | Default | Description |
|---|---|---|
| `conf_threshold` | `0.25` | Minimum confidence score to keep a detection |
| `VEHICLE_CLASS_IDS` | `{2, 3, 5, 7}` | COCO class IDs to detect (car, motorcycle, bus, truck) |

Tracker settings in `tracking/simple_tracker.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_MATCH_DIST` | `80` | Maximum pixel distance to match a detection to an existing track |
| `STALE_TIMEOUT` | `1.0` | Seconds before an unmatched track is removed |

## Troubleshooting

| Issue | Fix |
|---|---|
| `ValueError: Failed to load delegate from libedgetpu.so.1` | Coral USB not plugged in, or `libedgetpu` not installed. Run `test_delegate.py` to verify. |
| `No cameras available` | Check CSI ribbon cable and run `libcamera-hello` to verify the camera works. |
| Font warnings from Qt | Cosmetic only — does not affect functionality. Install `fonts-dejavu` if it bothers you. |
| Very low FPS | Make sure the Coral is on a USB 3.0 port. Increase `DETECT_EVERY_N_FRAMES` to skip frames. |
| Detections but no risk flags | Risk only triggers during red phase — press `r` to toggle. Check `MIN_SPEED` / `MAX_DIST` thresholds. |

## License

This project is provided as-is for educational and prototyping purposes.
