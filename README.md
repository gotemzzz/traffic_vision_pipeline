# TEAM 17 RED LIGHT RUNNER DETECTION

## SS26 ECE480 Capstone Project ~~ Traffic Infringement Unit

This repository showcases our addition to the project that was set up by the previous two groups.  
The full project has four units:

1. State Detection Unit  
2. Traffic Infringement Unit (**TFU**, this repo)  
3. Alarm Unit  
4. Power Supply

This repository focuses on the **Traffic Infringement Unit (TFU)**: real-time vehicle detection and red-light risk assessment on a Raspberry Pi 4 with a Google Coral Edge TPU.

---

## Pipeline Overview

The TFU pipeline runs as a staged processing flow:

| Stage | Module(s) | What it does | Output |
|---|---|---|---|
| 1. Capture | `picamera2` / image loader | Reads camera frames (`real_time` / `monitor`) or folder images (`images`) | Raw BGR frame |
| 2. Detect | `detector/coral_yolo_detector.py` or `detector/coral_tfod_detector.py` | Runs Edge TPU inference, confidence filtering, NMS, and duplicate suppression | Vehicle detections `(cx, cy, x, y, w, h)` |
| 3. Track | `tracking/simple_tracker.py` | Matches detections frame-to-frame, assigns stable IDs, smooths velocity vectors | Tracks with ID + speed + motion direction |
| 4. Risk Assessment | `risk/risk_logic.py` | Applies red-phase + direction-aware logic to determine risk/violation status | Per-track risk/violation state |
| 5. Output | `drawing/overlay.py` / `run/run_monitor.py` | Draw modes: overlays + labels. Monitor mode: alarm GPIO with hysteresis/debounce | Annotated frames or alarm signal |

### End-to-end flow (plain language)

1. **Capture** a frame from camera or input folder.  
2. **Detect** vehicles on the Edge TPU model.  
3. **Track** those vehicles across frames and estimate smoothed motion/speed.  
4. **Evaluate risk** only under red-phase logic with direction gating.  
5. **Emit output**:
   - `real_time` / `images`: draw visual annotations
   - `monitor`: run headless alarm logic (GPIO output optional, dry-run supported)

### Mode behavior summary

| Mode | Input | Display | Alarm GPIO | Light Sensor |
|---|---|---|---|---|
| `real_time` | Live camera | Yes | No | Optional (`--light-sensor`) |
| `images` | Folder images | Optional (`--real-time`) | No | Optional in `--real-time` |
| `animate` | Folder images | Yes (playback only) | No | No |
| `monitor` | Live camera | No (headless) | Optional (`--alarm-pin`) | **Required (always used)** |

---

## Modes

- `real_time` — Live camera with overlays and keyboard controls
- `images` — Batch folder processing (plus optional real-time playback mode)
- `animate` — Playback folder of frames
- `monitor` — Headless production mode (light sensor required, optional alarm GPIO, optional dry-run)

---

## Project Structure

```text
traffic_vision_pipeline/
├── main.py
├── run/
│   ├── run_real_time.py
│   ├── run_images.py
│   └── run_monitor.py
├── detector/
│   ├── coral_yolo_detector.py
│   └── coral_tfod_detector.py
├── tracking/
│   └── simple_tracker.py
├── risk/
│   └── risk_logic.py
├── drawing/
│   └── overlay.py
├── sensors/
│   └── light_sensor.py
├── models/
├── requirements.txt
├── test_delegate.py
└── README.md
```

---

## Hardware Requirements

| Component | Details |
|---|---|
| **SBC** | Raspberry Pi 4 (4GB+ recommended) |
| **AI accelerator** | [Google Coral USB Accelerator](https://coral.ai/products/accelerator/) |
| **Camera** | Pi-compatible CSI camera (tested with IMX296 mono) |
| **OS** | Raspberry Pi OS (64-bit / Bookworm recommended) |
| **Power** | 5V 3A USB-C |
| **Light sensor (optional in dev, required in monitor)** | Photoresistor divider or digital light sensor into GPIO input |
| **Alarm output (monitor mode optional)** | GPIO output pin to alarm relay/buzzer/controller |

---

## Software Prerequisites

- Python 3.11+ (Bookworm default)
- `libcamera` + `picamera2` (real_time/monitor)
- display/VNC only needed for visual modes (`real_time`, `images --real-time`, `animate`)

---

## Setup Guide (IMPORTANT — KEEP THIS)

## CRITICAL NOTE FOR RASPBERRY PI OS BOOKWORM (PYTHON 3.11)

Google’s official Coral apt packages are old (Python 3.9 / older TF tooling).  
Using those with modern Bookworm Python commonly causes segmentation faults (especially for models with custom ops).  
Use the community-maintained `feranick` builds below.

### 1) Install modern Edge TPU driver (feranick fork)

```bash
wget https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-max_16.0tf2.17.1-1.bookworm_arm64.deb
sudo dpkg -i libedgetpu1-max_16.0tf2.17.1-1.bookworm_arm64.deb
sudo ldconfig
```

### 2) Clone

```bash
git clone https://github.com/gotemzzz/traffic_vision_pipeline.git
cd traffic_vision_pipeline
```

### 3) Create venv + install requirements

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Install patched `tflite_runtime` and `pycoral` wheels (cp311 aarch64)

```bash
wget https://github.com/feranick/TFlite-builds/releases/download/v2.17.1/tflite_runtime-2.17.1-cp311-cp311-linux_aarch64.whl
wget https://github.com/feranick/pycoral/releases/download/2.0.3TF2.17.1/pycoral-2.0.3-cp311-cp311-linux_aarch64.whl

pip install tflite_runtime-2.17.1-cp311-cp311-linux_aarch64.whl
pip install pycoral-2.0.3-cp311-cp311-linux_aarch64.whl
```

### 5) Add model(s)

Place quantized Edge TPU model(s) in `models/`.

Example:
```text
models/yolov8n_full_integer_quant_edgetpu_192.tflite
```

Optional model export resources:
- https://docs.ultralytics.com/modes/export/
- https://github.com/jveitchmichaelis/edgetpu-yolo/
- https://gweb-coral-full.uc.r.appspot.com/models/object-detection/

### 6) Verify TPU delegate

```bash
python test_delegate.py
```

Expected:
```text
OK
```

---

## Usage

## A) real_time (live camera + overlays)

```bash
python main.py real_time
```

Example:
```bash
python main.py real_time \
  --conf 0.35 \
  --stop-line 0.65 \
  --detect-every 2 \
  --light-sensor --light-pin 17 \
  --approach-vx 0.0 --approach-vy 1.0
```

Controls:
- `q` quit
- `r` toggle red phase (manual mode only; disabled when light sensor is enabled)

### Save real-time frames for replay (NEW)

```bash
python main.py real_time \
  --save-frames \
  --save-dir output/real_run_01 \
  --save-every 1 \
  --save-prefix frame
```

Replay with:
```bash
python main.py animate -i output/real_run_01 --fps 10
```

---

## B) images (batch processing)

```bash
python main.py images -i frames/input -o frames/output
```

Simulated red:
```bash
python main.py images -i frames/input -o frames/output --red
```

Transitions (kept; not removed):
```bash
python main.py images \
  -i frames/input \
  -o frames/output \
  --transitions "green:0-50,red:51-150,green:151-end"
```

Real-time playback of input folder:
```bash
python main.py images -i frames/input --real-time --fps 10
```

With light sensor in images real-time mode:
```bash
python main.py images -i frames/input --real-time --light-sensor --light-pin 17
```

---

## C) animate

```bash
python main.py animate -i frames/output --fps 20
```

Controls:
- `Space` pause/resume
- `d` / `→` next frame
- `a` / `←` previous frame
- `q` / `ESC` quit

---

## D) monitor (headless production mode)

`monitor` is designed for deployment:
- no drawing/display
- **light sensor is required and always used**
- optional GPIO alarm output
- hysteresis/debounce
- dry-run support

### Dry run (recommended first)

```bash
python main.py monitor \
  --light-pin 17 \
  --alarm-pin 27 \
  --dry-run \
  --alarm-on-frames 3 \
  --alarm-off-frames 8
```

### Live alarm mode

```bash
python main.py monitor \
  --light-pin 17 \
  --alarm-pin 27 \
  --alarm-on-frames 3 \
  --alarm-off-frames 8 \
  --approach-vx 0.0 --approach-vy 1.0
```

---

## Configuration Reference

## real_time flags

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | yolov8n tflite | Edge TPU model path |
| `--conf` | `0.35` | detector confidence |
| `--width` | `640` | camera width |
| `--height` | `480` | camera height |
| `--stop-line` | `0.7` | stop line position (frame fraction) |
| `--detect-every` | `1` | detect every N frames |
| `--draw-every` | `1` | draw every N frames |
| `--light-sensor` | off | enable sensor gate |
| `--light-pin` | `17` | sensor GPIO pin |
| `--approach-vx` | `0.0` | approach vector x |
| `--approach-vy` | `1.0` | approach vector y |
| `--save-frames` | off | save processed real-time frames |
| `--save-dir` | `output/real_time_frames` | frame output folder |
| `--save-every` | `1` | save every N frames |
| `--save-prefix` | `frame` | output file prefix |

## images flags

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | required | input folder |
| `--output`, `-o` | required (unless `--real-time`) | output folder |
| `--model`, `-m` | yolov8n tflite | model path |
| `--conf` | `0.35` | confidence |
| `--red` | off | force red phase |
| `--transitions` | none | frame phase map |
| `--stop-line` | `0.7` | stop line fraction |
| `--no-track` | off | disable tracking |
| `--real-time` | off | display pipeline in real-time |
| `--animate` | off | animate outputs after processing |
| `--fps` | `10` | playback/assumed timing |
| `--light-sensor` | off | sensor gate for `--real-time` |
| `--light-pin` | `17` | sensor pin |
| `--approach-vx` | `0.0` | approach vector x |
| `--approach-vy` | `1.0` | approach vector y |

## monitor flags

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | yolov8n tflite | model path |
| `--conf` | `0.35` | confidence |
| `--width` | `640` | camera width |
| `--height` | `480` | camera height |
| `--stop-line` | `0.7` | stop line fraction |
| `--detect-every` | `1` | detect every N frames |
| `--light-pin` | `17` | **required sensor input** |
| `--alarm-pin` | none | GPIO output for alarm |
| `--dry-run` | off | no GPIO writes, log transitions only |
| `--alarm-on-frames` | `3` | hysteresis ON debounce |
| `--alarm-off-frames` | `8` | hysteresis OFF debounce |
| `--approach-vx` | `0.0` | approach vector x |
| `--approach-vy` | `1.0` | approach vector y |

---

## Implementation Notes / What Changed

- Direction-aware risk gating to reduce horizontal traffic false positives.
- Tracker now carries smoothed velocity vectors.
- Detector dedupe improved:
  - confidence filter
  - min box area filter
  - center-distance dedupe
  - IoU dedupe
- `monitor` mode added with hysteresis alarm logic.
- `monitor --dry-run` added for safe bench testing.
- `real_time` frame saving added for replay via `animate`.

---

## Light Sensor Logic

In `sensors/light_sensor.py`:

- GPIO `0` => RED (detection active)
- GPIO `1` => GREEN (detection gated off)

Adjust wiring/pull resistors accordingly.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Segmentation fault on model load/invoke | Mismatch between libedgetpu + tflite_runtime. Reinstall feranick driver/wheels exactly as above. |
| `Failed to load delegate libedgetpu.so.1` | Coral not detected / wrong install. Re-run `test_delegate.py`. |
| No camera | Verify CSI cable and run `libcamera-hello`. |
| Low FPS | Use USB 3.0 for Coral, increase `--detect-every`, reduce resolution. |
| Horizontal traffic still flagged | Tune `--approach-vx`/`--approach-vy`, verify stop line placement. |
| Alarm chatters | Increase `--alarm-on-frames` and `--alarm-off-frames`. |
| Too many boxes/duplicates | Raise `--conf`; tune detector dedupe constants if needed. |
| No risk flags | Ensure red phase is active (manual toggle or light sensor state). |

---

## Suggested Bring-Up Procedure

1. Validate TPU (`test_delegate.py`).
2. Run `real_time` with overlays; tune stop line and approach vector.
3. Validate with `real_time --save-frames`,~~

## License

This project is provided as-is for educational and prototyping purposes.
