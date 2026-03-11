import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# COCO classes that count as vehicles
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

class CoralYOLODetector:
    def __init__(self, model_path, conf_threshold=0.25):
        # Load TFLite model with Edge TPU delegate
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
        self.interpreter.allocate_tensors()

        self.conf_threshold = conf_threshold

        # Input & output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Model input size
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # Output dequantization parameters
        out_quant = self.output_details[0].get('quantization_parameters', {})
        self.out_scale = out_quant.get('scales', [1.0])[0]
        self.out_zero_point = out_quant.get('zero_points', [0])[0]

        # Ensure input type is INT8
        if self.input_details[0]['dtype'] != np.int8:
            raise ValueError(f"Expected INT8 model, got {self.input_details[0]['dtype']}")

        print(f"[MODEL] input: {self.input_details[0]['shape']} | "
              f"output: {self.output_details[0]['shape']} | "
              f"scale: {self.out_scale}, zp: {self.out_zero_point}")

    def detect(self, frame):
        """Run YOLOv8 detection on a single frame.

        Output tensor is [1, 84, 756] for 192x192 input:
          - 84 = 4 box values (cx, cy, w, h normalized) + 80 COCO class scores
          - 756 = anchor predictions
        After transpose we get [756, 84] — one row per anchor.
        """

        # Resize and convert BGR -> RGB
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert UINT8 -> INT8 [-128,127] safely
        img = img.astype(np.int16)
        img = img - 128
        img = np.clip(img, -128, 127)
        img = img.astype(np.int8)

        if len(self.input_details[0]['shape']) == 4:
            img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        # Get raw output and dequantize
        raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = self.out_scale * (raw_output.astype(np.float32) - self.out_zero_point)
        output_data = output_data[0]  # remove batch dim -> [84, 756]

        # Transpose so each row is one anchor: [756, 84]
        output_data = output_data.T

        # Split: first 4 = box coords, remaining 80 = class scores
        boxes = output_data[:, :4]          # (cx, cy, w, h) normalized 0-1
        class_scores = output_data[:, 4:]   # 80 COCO class scores

        # Best class per anchor
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)

        # Filter by confidence
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        h_orig, w_orig, _ = frame.shape

        detections = []
        for i in range(len(boxes)):
            cx_n, cy_n, bw_n, bh_n = boxes[i]
            score = scores[i]
            class_id = int(class_ids[i])

            # Optional: filter to vehicle classes only
            # Remove this check if you want to detect all 80 classes
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            # Convert normalized coords to original frame pixels
            cx_px = cx_n * w_orig
            cy_px = cy_n * h_orig
            bw_px = bw_n * w_orig
            bh_px = bh_n * h_orig

            x_min = int(cx_px - bw_px / 2)
            y_min = int(cy_px - bh_px / 2)
            x_max = int(cx_px + bw_px / 2)
            y_max = int(cy_px + bh_px / 2)

            # Clamp to frame bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w_orig, x_max)
            y_max = min(h_orig, y_max)

            w_box = x_max - x_min
            h_box = y_max - y_min

            if w_box <= 0 or h_box <= 0:
                continue

            cx_out = int((x_min + x_max) / 2)
            cy_out = int((y_min + y_max) / 2)

            detections.append([cx_out, cy_out, x_min, y_min, w_box, h_box])

        return detections
