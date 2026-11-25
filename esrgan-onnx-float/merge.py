# /// script
# requires-python = ">=3.13"
# dependencies = ["onnx"]
# ///

import onnx

input_path = "esrgan.onnx"
output_path = "esrgan_single.onnx"
model = onnx.load(input_path)
onnx.save(model, output_path)
