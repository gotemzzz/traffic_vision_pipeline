import math

MIN_SPEED = 28
MAX_DIST = 240
VIOLATION_DIST = -20

MIN_APPROACH_SPEED = 10.0
MIN_ALIGNMENT = 0.45      # hard direction gate (0=any direction, 1=perfectly aligned)
NEAR_LINE_DIST = 90


def _normalize(vx, vy):
    n = math.hypot(vx, vy)
    if n <= 1e-9:
        return 0.0, 1.0
    return vx / n, vy / n


def direction_alignment(vx, vy, approach_vx=0.0, approach_vy=1.0):
    """
    cosine similarity of motion vector vs approach vector:
      +1 = same direction
       0 = perpendicular
      -1 = opposite
    """
    vmag = math.hypot(vx, vy)
    if vmag <= 1e-9:
        return 0.0
    ax, ay = _normalize(approach_vx, approach_vy)
    return (vx * ax + vy * ay) / vmag


def approach_speed(vx, vy, approach_vx=0.0, approach_vy=1.0):
    ax, ay = _normalize(approach_vx, approach_vy)
    return vx * ax + vy * ay


def evaluate_risk(
    red_phase,
    cy,
    stop_line,
    speed,
    violation_history=None,
    vx=0.0,
    vy=0.0,
    approach_vx=0.0,
    approach_vy=1.0,
):
    if not red_phase:
        return False

    dist = stop_line - cy

    if violation_history:
        return True

    # immediate risk if clearly crossed on red
    if dist <= VIOLATION_DIST:
        return True

    # ignore far vehicles
    if dist >= MAX_DIST:
        return False

    align = direction_alignment(vx, vy, approach_vx, approach_vy)
    apd = approach_speed(vx, vy, approach_vx, approach_vy)

    # Hard gate: must be moving generally toward stop line direction
    if align < MIN_ALIGNMENT:
        return False

    # Aggressive near-line catch
    if dist <= NEAR_LINE_DIST and (apd >= MIN_APPROACH_SPEED or speed >= MIN_SPEED):
        return True

    # Mid-range catch
    if apd >= (MIN_APPROACH_SPEED + 6.0):
        return True

    return False


def update_violation_status(
    track,
    stop_line,
    red_phase,
    approach_vx=0.0,
    approach_vy=1.0,
):
    # new format
    if len(track) >= 12:
        tid, cx, cy, x, y, w, h, speed, violation, vx, vy, matched = track[:12]
    # older new-ish format
    elif len(track) >= 11:
        tid, cx, cy, x, y, w, h, speed, violation, vx, vy = track[:11]
        matched = True
    else:
        # oldest compatibility
        tid, cx, cy, x, y, w, h, speed, violation = track[:9]
        vx, vy = 0.0, speed
        matched = True

    if not red_phase:
        return False

    if violation:
        return True

    # if not matched this frame, don't newly trigger violation from stale state
    if not matched:
        return False

    dist = stop_line - cy
    align = direction_alignment(vx, vy, approach_vx, approach_vy)
    apd = approach_speed(vx, vy, approach_vx, approach_vy)

    # crossing violation only if direction supports it
    if dist <= VIOLATION_DIST and align >= MIN_ALIGNMENT and apd >= MIN_APPROACH_SPEED:
        return True

    # near-line "runner latch"
    if dist <= 20 and align >= MIN_ALIGNMENT and (apd >= MIN_APPROACH_SPEED or speed >= MIN_SPEED):
        return True

    return False
