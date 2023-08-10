from furiosa.models.vision import SSDMobileNet
from furiosa.runtime.sync import create_runner

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet("Rust")
with create_runner(mobilenet.model_source()) as runner:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = runner.run(inputs)
    mobilenet.postprocess(outputs, contexts[0])
