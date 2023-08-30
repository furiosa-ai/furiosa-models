from furiosa.models.vision import SSDMobileNet
from furiosa.quantizer import quantize
from furiosa.runtime.sync import create_runner

compiler_config = {"lower_tabulated_dequantize": True}

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet()
onnx_model: bytes = mobilenet.origin
calib_range: dict = mobilenet.tensor_name_to_range

# See https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.quantizer.html#furiosa.quantizer.quantize
# for more details
quantized_onnx = quantize(onnx_model, calib_range)

with create_runner(quantized_onnx, compiler_config=compiler_config) as runner:
    # Models in the Model Zoo have built-in optimizations that, by default,
    # bypass normalization, quantization, and type conversion. If you compile
    # and utilize these models without employing these optimizations, it's
    # necessary to set up preprocessing steps to incorporate normalization and
    # type casting. To accomplish this, you should introduce an extra parameter,
    # `with_scaling=True`.
    inputs, contexts = mobilenet.preprocess(image, with_scaling=True)
    outputs = runner.run(inputs)
    mobilenet.postprocess(outputs, contexts[0])
