import cv2


def draw_tracks(frame, tracks, risk_fn, red_phase, stop_line_y):
    """Draw bounding boxes, labels, stop line, and phase onto a frame.

    Args:
        frame:       BGR image (modified in-place)
        tracks:      list of (tid, cx, cy, x, y, w, h, speed)
        risk_fn:     callable(red_phase, cy, stop_line_y, speed) -> bool
        red_phase:   whether the light is red
        stop_line_y: pixel y-coordinate of the stop line
    """
    h, w = frame.shape[:2]

    for track in tracks:
        tid, cx, cy, x, y, w_box, h_box, speed = track
        risk = risk_fn(red_phase, cy, stop_line_y, speed)
        color = (0, 0, 255) if risk else (0, 255, 0)

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

        # Label with background for readability
        label = f"ID {tid} {int(speed)}px/s" + (" RISK" if risk else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y - 6, th + 4)  # keep label on-screen

        # Dark background rectangle behind text
        cv2.rectangle(frame,
                      (x, label_y - th - 4),
                      (x + tw + 4, label_y + baseline),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, label, (x + 2, label_y - 2),
                    font, font_scale, color, thickness, cv2.LINE_AA)

    # Stop line
    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 255, 255), 2)

    # Phase indicator
    phase = "RED" if red_phase else "GREEN"
    phase_color = (0, 0, 255) if red_phase else (0, 200, 0)
    cv2.putText(frame, phase, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, phase_color, 2, cv2.LINE_AA)
