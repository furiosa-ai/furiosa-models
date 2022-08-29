# FIXME: This is a temporary test script to check rust binding has imported
from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision.ssd_mobilenet import CLASSES, preprocess, RustPostProcessor
from furiosa.runtime import session, tensor
import numpy as np


def test_rust_post_processor():
    test_image_path = "/home/hyunsik/Downloads/M22_3548_fine.jpeg"

    assert len(CLASSES) == 92, f"Classes is 92, but {len(CLASSES)}"
    true_bbox = np.array([[187.30786, 88.035324, 950.99646, 743.3290239999999]], dtype=np.float32)
    true_confidence = np.array([0.97390455], dtype=np.float32)
    # For batch mode test, simply read two identical images.
    batch_pre_image, batch_preproc_param = preprocess([test_image_path, test_image_path])

    model = SSDMobileNet()
    processor = RustPostProcessor(model)

    with session.create(model.enf) as sess:
        #print(f"input_num: {sess.input_num}")

        for idx, input in enumerate(sess.outputs()):
            print(f"{idx} - {input.shape}")

        outputs = sess.run(batch_pre_image).numpy()
        results = processor.eval(outputs)
        for res in results:
            print(res)

