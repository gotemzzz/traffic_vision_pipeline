import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# COCO classes that count as vehicles
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# NMS parameters
NMS_IOU_THRESHOLD = 0.5


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

        print(f"[MODEL] YOLO Loaded | input: {self.input_details[0]['shape']} | "
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
        boxes_norm = output_data[:, :4]       # (cx, cy, w, h) normalized 0-1
        class_scores = output_data[:, 4:]     # 80 COCO class scores

        # Best class per anchor
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)

        # Filter by confidence
        mask = scores > self.conf_threshold
        boxes_norm = boxes_norm[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_norm) == 0:
            return []

        h_orig, w_orig, _ = frame.shape

        # Convert to pixel coordinates for NMS
        # boxes_norm is (cx, cy, w, h) normalized
        cx_px = boxes_norm[:, 0] * w_orig
        cy_px = boxes_norm[:, 1] * h_orig
        bw_px = boxes_norm[:, 2] * w_orig
        bh_px = boxes_norm[:, 3] * h_orig

        x1 = (cx_px - bw_px / 2).astype(int)
        y1 = (cy_px - bh_px / 2).astype(int)
        x2 = (cx_px + bw_px / 2).astype(int)
        y2 = (cy_px + bh_px / 2).astype(int)

        # Clamp
        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)

        # OpenCV NMS expects (x, y, w, h) as a list of lists, and scores as a list
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_scores = scores.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores,
                                   self.conf_threshold, NMS_IOU_THRESHOLD)

        if len(indices) == 0:
            return []

        # Flatten — NMSBoxes returns [[0],[1],[2]] in some OpenCV versions
        if isinstance(indices[0], (list, np.ndarray)):
            indices = [i[0] for i in indices]

        detections = []
        for i in indices:
            class_id = int(class_ids[i])

            # Filter to vehicle classes only
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            bx, by, bw, bh = nms_boxes[i]

            if bw <= 0 or bh <= 0:
                continue

            cx_out = int(bx + bw / 2)
            cy_out = int(by + bh / 2)

            detections.append([cx_out, cy_out, int(bx), int(by), int(bw), int(bh)])

        return detections
