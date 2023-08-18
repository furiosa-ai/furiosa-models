from furiosa.models.vision import EfficientNetB0
from furiosa.runtime.sync import create_runner

image = "tests/assets/cat.jpg"

effnetb0 = EfficientNetB0()
with create_runner(effnetb0.model_source()) as runner:
    inputs, _ = effnetb0.preprocess(image)
    outputs = runner.run(inputs)
    effnetb0.postprocess(outputs)
