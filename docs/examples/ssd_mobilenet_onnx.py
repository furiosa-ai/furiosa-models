import yaml

from furiosa.models.vision import SSDMobileNet
from furiosa.quantizer import quantize
from furiosa.runtime import session

compiler_config = {"lower_tabulated_dequantize": True}

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet.load()
onnx_model: bytes = mobilenet.source
calib_range: dict = yaml.full_load(mobilenet.calib_yaml)

# See https://furiosa-ai.github.io/docs/latest/en/api/python/furiosa.quantizer.html#furiosa.quantizer.quantize
# for more details
dfg = quantize(onnx_model, calib_range, with_quantize=False)

with session.create(dfg, compiler_config=compiler_config) as sess:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = sess.run(inputs).numpy()
    mobilenet.postprocess(outputs, contexts[0])
