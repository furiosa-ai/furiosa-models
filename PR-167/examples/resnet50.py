from furiosa.models.vision import ResNet50
from furiosa.runtime.sync import create_runner

image = "tests/assets/cat.jpg"

resnet50 = ResNet50()
with create_runner(resnet50.model_source()) as runner:
    inputs, _ = resnet50.preprocess(image)
    outputs = runner.run(inputs)
    resnet50.postprocess(outputs)
