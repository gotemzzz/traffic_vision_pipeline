import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Standard TF Object Detection API vehicle classes (COCO with 90 classes)
# Depending on the specific model, it might be 0-indexed or 1-indexed.
# YOLO uses 2=car, 3=motorcycle, 5=bus, 7=truck. TFOD often uses 3, 4, 6, 8.
# We include both sets to be safe and ensure all vehicles are caught.
TFOD_VEHICLE_CLASS_IDS = {2, 3, 5, 7, 3, 4, 6, 8}

class CoralTFODDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        # 1. THE SEGFAULT FIX: We MUST store the delegate in an instance variable
        # so Python's garbage collector doesn't destroy it.
        self._delegate = load_delegate('libedgetpu.so.1')
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[self._delegate]
        )
        self.interpreter.allocate_tensors()

        self.conf_threshold = conf_threshold

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Model input size
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.input_dtype = self.input_details[0]['dtype']

        print(f"[MODEL] TFOD Loaded | input: {self.input_details[0]['shape']} | dtype: {self.input_dtype}")

    def detect(self, frame):
        """Run EfficientDet/MobileNet SSD detection on a single frame."""
        # Resize and convert BGR -> RGB
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle different quantization inputs (usually UINT8 for Coral EfficientDet)
        if self.input_dtype == np.uint8:
            img = img.astype(np.uint8)
        elif self.input_dtype == np.int8:
            img = img.astype(np.int16) - 128
            img = np.clip(img, -128, 127).astype(np.int8)

        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        # TFOD outputs 4 tensors. We map them explicitly by their expected order:
        # 0: boxes [1, N, 4] -> (ymin, xmin, ymax, xmax)
        # 1: classes [1, N]
        # 2: scores [1, N]
        # 3: count [1] -> Number of valid detections
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        count = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

        if count == 0:
            return []

        h_orig, w_orig, _ = frame.shape
        detections = []

        for i in range(count):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            class_id = int(classes[i])
            if class_id not in TFOD_VEHICLE_CLASS_IDS:
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            # Convert normalized coordinates to pixel coordinates
            x1 = int(max(0, xmin * w_orig))
            y1 = int(max(0, ymin * h_orig))
            x2 = int(min(w_orig, xmax * w_orig))
            y2 = int(min(h_orig, ymax * h_orig))

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            cx = x1 + bw // 2
            cy = y1 + bh // 2

            # Output matches what SimpleTracker expects: (cx, cy, bx, by, bw, bh)
            detections.append([cx, cy, x1, y1, bw, bh])

        return detections
