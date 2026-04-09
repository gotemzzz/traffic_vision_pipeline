import cv2


def draw_tracks(frame, tracks, risk_fn, red_phase, stop_line_y, approach_vx=0.0, approach_vy=1.0):
    """
    Draw only matched tracks to avoid ghost trails.
    Supports track formats:
      old: (..., speed, violation)
      new: (..., speed, violation, vx, vy)
      newest: (..., speed, violation, vx, vy, matched)
    """
    h, w = frame.shape[:2]

    for track in tracks:
        if len(track) >= 12:
            tid, cx, cy, x, y, w_box, h_box, speed, violation, vx, vy, matched = track[:12]
        elif len(track) >= 11:
            tid, cx, cy, x, y, w_box, h_box, speed, violation, vx, vy = track[:11]
            matched = True
        else:
            tid, cx, cy, x, y, w_box, h_box, speed, violation = track[:9]
            vx, vy = 0.0, 0.0
            matched = True

        # IMPORTANT: skip stale/unmatched tracks to prevent path-like trail boxes
        if not matched:
            continue

        risk = risk_fn(
            red_phase,
            cy,
            stop_line_y,
            speed,
            violation,
            vx=vx,
            vy=vy,
            approach_vx=approach_vx,
            approach_vy=approach_vy,
        )

        color = (0, 0, 255) if risk else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

        # velocity arrow
        arrow_scale = 0.08
        ex = int(cx + vx * arrow_scale)
        ey = int(cy + vy * arrow_scale)
        cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), (255, 200, 0), 2, tipLength=0.25)

        label = f"ID {tid} v={int(speed)}" + (" VIOLATION" if risk else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y - 6, th + 4)

        cv2.rectangle(
            frame,
            (x, label_y - th - 4),
            (x + tw + 4, label_y + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(frame, label, (x + 2, label_y - 2),
                    font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 255, 255), 2)

    phase = "RED" if red_phase else "GREEN"
    phase_color = (0, 0, 255) if red_phase else (0, 200, 0)
    cv2.putText(frame, phase, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, phase_color, 2, cv2.LINE_AA)
