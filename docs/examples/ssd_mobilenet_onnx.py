from furiosa.models.vision import SSDMobileNet
from furiosa.runtime import session

images = ["tests/assets/cat.jpg", "tests/assets/cat.jpg"]

mobilenet = SSDMobileNet.load()
with session.create(mobilenet.source, batch_size=2) as sess:
    inputs, contexts = mobilenet.preprocess(images)
    outputs = sess.run(inputs).numpy()
    mobilenet.postprocess(outputs, contexts=contexts)
