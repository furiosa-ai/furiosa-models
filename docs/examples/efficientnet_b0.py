from furiosa.models.vision import EfficientNetB0
from furiosa.runtime import session

image = "tests/assets/cat.jpg"

effnetb0 = EfficientNetB0()
with session.create(effnetb0.model_source()) as sess:
    inputs, _ = effnetb0.preprocess(image)
    outputs = sess.run(inputs).numpy()
    effnetb0.postprocess(outputs)
