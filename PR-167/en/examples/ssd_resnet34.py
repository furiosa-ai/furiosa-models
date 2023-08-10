from furiosa.models.vision import SSDResNet34
from furiosa.runtime import session

resnet34 = SSDResNet34("Python")

with session.create(resnet34.model_source()) as sess:
    image, contexts = resnet34.preprocess(["tests/assets/cat.jpg"])
    output = sess.run(image)
    resnet34.postprocess(output, contexts=contexts)
