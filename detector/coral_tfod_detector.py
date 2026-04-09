import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# COCO vehicle classes (typical TFOD indexing may vary by model)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

CENTER_DEDUP_PX = 24
MIN_BOX_AREA = 500
IOU_DEDUP_THRESHOLD = 0.5


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class CoralTFODDetector:
    def __init__(self, model_path, conf_threshold=0.35):
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
        self.interpreter.allocate_tensors()

        self.conf_threshold = conf_threshold

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]

        print(f"[MODEL] TFOD Loaded | input: {self.input_details[0]['shape']}")

    def detect(self, frame):
        # Preprocess
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_dtype = self.input_details[0]["dtype"]
        if input_dtype == np.uint8:
            input_data = np.expand_dims(img_rgb.astype(np.uint8), axis=0)
        elif input_dtype == np.int8:
            # int8 quantized input
            input_data = img_rgb.astype(np.int16) - 128
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
        else:
            # float fallback
            input_data = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Typical TFOD output tensors:
        # boxes:   [1, N, 4] normalized ymin, xmin, ymax, xmax
        # classes: [1, N]
        # scores:  [1, N]
        # count:   [1]
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        h, w, _ = frame.shape
        candidates = []

        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.conf_threshold:
                continue

            class_id = int(classes[i])
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            x1 = int(max(0, min(w, xmin * w)))
            y1 = int(max(0, min(h, ymin * h)))
            x2 = int(max(0, min(w, xmax * w)))
            y2 = int(max(0, min(h, ymax * h)))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if (bw * bh) < MIN_BOX_AREA:
                continue

            cx = x1 + bw // 2
            cy = y1 + bh // 2

            candidates.append([cx, cy, x1, y1, bw, bh, score])

        if not candidates:
            return []

        # Sort by score desc
        candidates.sort(key=lambda d: d[6], reverse=True)

        # Pass 1: center dedupe
        kept = []
        for det in candidates:
            cx, cy = det[0], det[1]
            too_close = False
            for k in kept:
                if (cx - k[0]) ** 2 + (cy - k[1]) ** 2 <= CENTER_DEDUP_PX ** 2:
                    too_close = True
                    break
            if not too_close:
                kept.append(det)

        # Pass 2: IoU dedupe
        final = []
        for det in kept:
            db = (det[2], det[3], det[4], det[5])
            dup = False
            for f in final:
                fb = (f[2], f[3], f[4], f[5])
                if _iou_xywh(db, fb) > IOU_DEDUP_THRESHOLD:
                    dup = True
                    break
            if not dup:
                final.append(det)

        return [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in final]
