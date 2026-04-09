import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

NMS_IOU_THRESHOLD = 0.40
CENTER_DEDUP_PX = 24
MIN_BOX_AREA = 500


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
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


class CoralYOLODetector:
    def __init__(self, model_path, conf_threshold=0.35):
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
        self.interpreter.allocate_tensors()

        self.conf_threshold = conf_threshold
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

        out_quant = self.output_details[0].get('quantization_parameters', {})
        self.out_scale = out_quant.get('scales', [1.0])[0]
        self.out_zero_point = out_quant.get('zero_points', [0])[0]

        if self.input_details[0]['dtype'] != np.int8:
            raise ValueError(f"Expected INT8 model, got {self.input_details[0]['dtype']}")

        print(f"[MODEL] YOLO Loaded | input: {self.input_details[0]['shape']} | "
              f"output: {self.output_details[0]['shape']} | "
              f"scale: {self.out_scale}, zp: {self.out_zero_point}")

    def detect(self, frame):
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.int16)
        img = img - 128
        img = np.clip(img, -128, 127)
        img = img.astype(np.int8)

        if len(self.input_details[0]['shape']) == 4:
            img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = self.out_scale * (raw_output.astype(np.float32) - self.out_zero_point)
        output_data = output_data[0].T  # [N, 84]

        boxes_norm = output_data[:, :4]
        class_scores = output_data[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)

        mask = scores > self.conf_threshold
        boxes_norm = boxes_norm[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_norm) == 0:
            return []

        h_orig, w_orig, _ = frame.shape

        cx_px = boxes_norm[:, 0] * w_orig
        cy_px = boxes_norm[:, 1] * h_orig
        bw_px = boxes_norm[:, 2] * w_orig
        bh_px = boxes_norm[:, 3] * h_orig

        x1 = (cx_px - bw_px / 2).astype(int)
        y1 = (cy_px - bh_px / 2).astype(int)
        x2 = (cx_px + bw_px / 2).astype(int)
        y2 = (cy_px + bh_px / 2).astype(int)

        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_scores = scores.tolist()

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            nms_scores,
            self.conf_threshold,
            NMS_IOU_THRESHOLD
        )

        if len(indices) == 0:
            return []

        if isinstance(indices[0], (list, np.ndarray)):
            indices = [i[0] for i in indices]

        # Build candidates
        candidates = []
        for i in indices:
            class_id = int(class_ids[i])
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            bx, by, bw, bh = nms_boxes[i]
            if bw <= 0 or bh <= 0:
                continue
            if (bw * bh) < MIN_BOX_AREA:
                continue

            cx_out = int(bx + bw / 2)
            cy_out = int(by + bh / 2)
            candidates.append([cx_out, cy_out, int(bx), int(by), int(bw), int(bh), float(scores[i])])

        if not candidates:
            return []

        # Sort by confidence desc
        candidates.sort(key=lambda d: d[6], reverse=True)

        # pass 1: center dedupe
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

        # pass 2: IoU dedupe among kept
        final = []
        for det in kept:
            bbox = (det[2], det[3], det[4], det[5])
            duplicate = False
            for f in final:
                fbox = (f[2], f[3], f[4], f[5])
                if _iou_xywh(bbox, fbox) > 0.5:
                    duplicate = True
                    break
            if not duplicate:
                final.append(det)

        return [[d[0], d[1], d[2], d[3], d[4], d[5]] for d in final]
