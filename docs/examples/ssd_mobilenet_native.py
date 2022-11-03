from furiosa.models.vision import SSDMobileNet
from furiosa.runtime import session

mobilenet = SSDMobileNet.load(use_native=True)

image = ["tests/assets/cat.jpg"]

with session.create(mobilenet) as sess:
    inputs, contexts = mobilenet.preprocess(image)
    outputs = sess.run(inputs).numpy()
    outputs = mobilenet.postprocess(outputs, contexts[0])
    print(outputs)
