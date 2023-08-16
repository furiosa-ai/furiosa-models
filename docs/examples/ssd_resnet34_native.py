from furiosa.models.vision import SSDResNet34
from furiosa.models.types import Platform
from furiosa.runtime.sync import create_runner

resnet34 = SSDResNet34(postprocessor_type=Platform.RUST)

with create_runner(resnet34.model_source()) as runner:
    image, contexts = resnet34.preprocess(["tests/assets/cat.jpg"])
    output = runner.run(image)
    resnet34.postprocessor(output, contexts=contexts[0])
