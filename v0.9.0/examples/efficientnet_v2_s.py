from furiosa.models.vision import EfficientNetV2s
from furiosa.runtime import session

image = "tests/assets/cat.jpg"

effnetv2s = EfficientNetV2s.load()
with session.create(effnetv2s.enf) as sess:
    inputs, _ = effnetv2s.preprocess(image)
    outputs = sess.run(inputs).numpy()
    effnetv2s.postprocess(outputs)
