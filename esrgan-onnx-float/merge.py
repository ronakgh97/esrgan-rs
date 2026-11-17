import onnx

input_path = "esrgan.onnx"
output_path = "esrgan_single.onnx"
model = onnx.load(input_path)
onnx.save(model, output_path)
