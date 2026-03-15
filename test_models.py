from tflite_runtime.interpreter import Interpreter, load_delegate

def test_model(name, path):
    print(f"\n--- Testing {name} ---")
    try:
        interpreter = Interpreter(
            model_path=path,
            experimental_delegates=[load_delegate('libedgetpu.so.1')]
        )
        interpreter.allocate_tensors()
        print("SUCCESS: Tensors allocated on Edge TPU!")
    except Exception as e:
        print(f"FAILED: {e}")

# Test both models
test_model("YOLO", "models/yolov8n_full_integer_quant_edgetpu_192.tflite")
test_model("EfficientDet", "models/efficientdet_lite1_384_ptq_edgetpu.tflite")
test_model("MobileNet", "models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")

