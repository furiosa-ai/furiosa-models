import yaml

from furiosa.models.vision import SSDMobileNet
from furiosa.quantizer import quantize
from furiosa.runtime import session

compiler_config = {"lower_tabulated_dequantize": True}

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet()
onnx_model: bytes = mobilenet.origin
calib_range: dict = mobilenet.tensor_name_to_range

# See https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.quantizer.html#furiosa.quantizer.quantize
# for more details
quantized_onnx = quantize(onnx_model, calib_range)

with session.create(quantized_onnx, compiler_config=compiler_config) as sess:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = sess.run(inputs).numpy()
    mobilenet.postprocess(outputs, contexts[0])
