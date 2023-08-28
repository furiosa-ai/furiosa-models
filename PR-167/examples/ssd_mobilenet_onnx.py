from furiosa.models.vision import SSDMobileNet
from furiosa.quantizer import quantize
from furiosa.runtime.sync import create_runner

compiler_config = {"lower_tabulated_dequantize": True}

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet()
onnx_model: bytes = mobilenet.origin
tensor_name_to_range: dict = mobilenet.tensor_name_to_range

# See https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.quantizer.html#furiosa.quantizer.quantize
# for more details
quantized_onnx = quantize(onnx_model, tensor_name_to_range)

with create_runner(quantized_onnx, compiler_config=compiler_config) as runner:
    inputs, contexts = mobilenet.preprocess(image, with_scaling=True)
    outputs = runner.run(inputs)
    mobilenet.postprocess(outputs, contexts[0])
