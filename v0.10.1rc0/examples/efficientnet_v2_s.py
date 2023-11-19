from furiosa.models.vision import EfficientNetV2s
from furiosa.runtime.sync import create_runner

image = "tests/assets/cat.jpg"

effnetv2s = EfficientNetV2s()
with create_runner(effnetv2s.model_source()) as runner:
    inputs, _ = effnetv2s.preprocess(image)
    outputs = runner.run(inputs)
    effnetv2s.postprocess(outputs)
