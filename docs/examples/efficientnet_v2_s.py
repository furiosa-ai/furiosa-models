from furiosa.models.vision import EfficientNetV2s
from furiosa.runtime import session

image = "tests/assets/cat.jpg"

effnetv2s = EfficientNetV2s()
with session.create(effnetv2s.model_source()) as sess:
    inputs, _ = effnetv2s.preprocess(image)
    outputs = sess.run(inputs).numpy()
    effnetv2s.postprocess(outputs)
