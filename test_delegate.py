import tflite_runtime.interpreter as tflite
tflite.load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1")
print("OK")
