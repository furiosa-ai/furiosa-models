from furiosa.models.vision import SSDMobileNet
from furiosa.runtime import session

image = ["tests/assets/cat.jpg"]

mobilenet = SSDMobileNet()
with session.create(mobilenet.model_source()) as sess:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = sess.run(inputs).numpy()
    mobilenet.postprocess(outputs, contexts[0])
