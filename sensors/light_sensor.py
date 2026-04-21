import RPi.GPIO as GPIO
import threading
import time


class LightSensor:
    """
    Digital light gate sensor.

    Convention:
      active_low=True (default): GPIO reads 0 -> RED phase (run detection)
      active_low=False: GPIO reads 1 -> RED phase (run detection)
    """

    def __init__(self, pin=17, poll_interval=0.05, active_low=True):
        self.pin = pin
        self.poll_interval = poll_interval
        self.active_low = active_low
        self.running = False
        self._is_red = False
        self._lock = threading.Lock()

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)

    def start(self):
        if self.running:
            return
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        mode = "active LOW" if self.active_low else "active HIGH"
        print(f"[SENSOR] Light sensor started on GPIO {self.pin} ({mode})")

    def stop(self):
        self.running = False
        # cleanup only this pin so we don't affect other GPIO users
        GPIO.cleanup(self.pin)
        print("[SENSOR] Light sensor stopped")

    def is_red(self):
        with self._lock:
            return self._is_red

    def _loop(self):
        while self.running:
            val = GPIO.input(self.pin)
            with self._lock:
                if self.active_low:
                    self._is_red = (val == 0)
                else:
                    self._is_red = (val == 1)
            time.sleep(self.poll_interval)
