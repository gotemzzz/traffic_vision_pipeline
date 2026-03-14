import tflite_runtime.interpreter as tflite

try:
	tflite.load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1")
	print("OK")
except Exception as e:
	print("Failed to load libedgetpu.so.1, make sure that your TPU is plugged in and recieving power")
