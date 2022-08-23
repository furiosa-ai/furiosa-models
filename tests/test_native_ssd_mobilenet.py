# FIXME: This is a temporary test script to check rust binding has imported
from furiosa.models.vision.native import ssd_mobilenet
from furiosa.runtime import session, tensor
import numpy as np


def test_rust_post_processor():
    with open("models/mlcommons_ssd_mobilenet_v1_int8.onnx_truncated.dfg", "rb") as file:
        dfg = file.read();
        processor = ssd_mobilenet.RustPostprocessor(dfg)

        with session.create("models/mlcommons_ssd_mobilenet_v1_int8.onnx_truncated.enf") as sess:
            #print(f"input_num: {sess.input_num}")

            outputs = sess.run(tensor.rand(sess.input(0))).numpy()
            #assert len(outputs) == 12

            processor.process(np.array([[123], [123]], np.int8))
