MIN_SPEED = 80
MAX_DIST = 200


def evaluate_risk(red_phase,cy,stop_line,speed):

    if not red_phase:
        return False

    dist = stop_line - cy

    if 0 < dist < MAX_DIST and speed > MIN_SPEED:
        return True

    return False
