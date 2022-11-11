from furiosa.models.vision import SSDMobileNet
from furiosa.runtime import session

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet.load()
with session.create(mobilenet) as sess:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = sess.run(inputs).numpy()
    mobilenet.postprocess(outputs, contexts)
