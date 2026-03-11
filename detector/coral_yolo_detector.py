import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

class CoralYOLODetector:
    def __init__(self, model_path):
        # Load TFLite model with Edge TPU delegate
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
        self.interpreter.allocate_tensors()

        # Input & output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Model input size
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        # Ensure input type is INT8
        if self.input_details[0]['dtype'] != np.int8:
            raise ValueError(f"Expected INT8 model, got {self.input_details[0]['dtype']}")

    def detect(self, frame):
        """Run YOLO detection on a single frame"""

        # Resize and convert BGR -> RGB
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert UINT8 -> INT8 [-128,127] safely
        img = img.astype(np.int16)       # promote to avoid overflow
        img = img - 128
        img = np.clip(img, -128, 127)    # clamp to INT8 range
        img = img.astype(np.int8)

        # Add batch dimension if needed
        if len(self.input_details[0]['shape']) == 4:
            img = np.expand_dims(img, axis=0)

        # Feed image to interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        # Edge TPU YOLO models usually output a single tensor [1,N,6]
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # [N,6]

        h_orig, w_orig, _ = frame.shape
        detections = []

        for det in output_data:
            x_min, y_min, x_max, y_max, score, class_id = det[:6]

            if score < 0.3:
                continue

            # Scale box to original frame
            x_min = int(x_min * w_orig)
            y_min = int(y_min * h_orig)
            x_max = int(x_max * w_orig)
            y_max = int(y_max * h_orig)
            w_box = x_max - x_min
            h_box = y_max - y_min

            detections.append([x_min, y_min, w_box, h_box, float(score), int(class_id)])

        return detections
