from furiosa.models.vision import SSDResNet34
from furiosa.runtime import session

resnet34 = SSDResNet34.load()

with session.create(resnet34) as sess:
    image, contexts = resnet34.preprocess(["tests/assets/cat.jpg"])
    output = sess.run(image).numpy()
    resnet34.postprocess(output, contexts=contexts)
