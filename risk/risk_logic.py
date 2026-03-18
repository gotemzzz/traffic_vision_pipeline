MIN_SPEED = 80
MAX_DIST = 200
VIOLATION_DIST = -20  # How far past the line counts as violation (negative = past)


def evaluate_risk(red_phase, cy, stop_line, speed, violation_history=None):
    """
    Evaluate red light violation risk.
    
    Args:
        red_phase: Whether the light is currently red
        cy: Vehicle center Y position
        stop_line: Y coordinate of stop line
        speed: Current speed in pixels/frame
        violation_history: Optional bool indicating if vehicle already violated
    
    Returns:
        bool: True if vehicle is in violation
    """
    if not red_phase:
        return False

    dist = stop_line - cy

    # **Primary violation**: Car has passed the stop line on red
    # Once past, it stays in violation regardless of speed
    if violation_history:
        return True
    
    # **Secondary violation**: Car approaching/crossing with high speed
    # 0 < dist < MAX_DIST means approaching from above
    # dist < 0 means already past the line
    if dist < MAX_DIST and speed > MIN_SPEED:
        return True

    return False


def update_violation_status(track, stop_line, red_phase):
    """
    Update violation status for a track.
    Called once per frame for each tracked vehicle.
    
    Args:
        track: tuple (tid, cx, cy, x, y, w, h, speed, violation)
        stop_line: Y coordinate of stop line
        red_phase: Whether light is red
    
    Returns:
        bool: Updated violation status
    """
    tid, cx, cy, x, y, w, h, speed, violation = track
    
    if not red_phase:
        return False  # Reset if light turns green
    
    dist = stop_line - cy
    
    # If already violated, stay violated
    if violation:
        return True
    
    # If passing the line on red, mark as violated
    # Vehicle crosses if dist goes from positive to negative or ≤ 0
    if dist <= VIOLATION_DIST:
        return True
    
    return False
