from furiosa.models.vision import SSDMobileNet
from furiosa.models.types import Platform
from furiosa.runtime.sync import create_runner

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet(postprocessor_type=Platform.RUST)
with create_runner(mobilenet.model_source()) as runner:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = runner.run(inputs)
    mobilenet.postprocess(outputs, contexts[0])
