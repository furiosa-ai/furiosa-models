from furiosa.models.vision import ResNet50
from furiosa.runtime import session

image = "tests/assets/cat.jpg"

resnet50 = ResNet50()
with session.create(resnet50.model_source()) as sess:
    inputs, _ = resnet50.preprocess(image)
    outputs = sess.run(inputs)
    resnet50.postprocess(outputs)
