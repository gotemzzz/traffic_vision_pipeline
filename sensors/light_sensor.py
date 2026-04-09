import RPi.GPIO as GPIO
import threading
import time


class LightSensor:
    """
    Digital light gate sensor.

    Convention used here:
      GPIO reads 0 -> RED phase (run detection)
      GPIO reads 1 -> GREEN phase (skip detection)
    """

    def __init__(self, pin=17, poll_interval=0.05):
        self.pin = pin
        self.poll_interval = poll_interval
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
        print(f"[SENSOR] Light sensor started on GPIO {self.pin}")

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
                self._is_red = (val == 0)
            time.sleep(self.poll_interval)
