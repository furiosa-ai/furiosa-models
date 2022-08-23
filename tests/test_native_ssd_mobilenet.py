# FIXME: This is a temporary test script to check rust binding has imported
from furiosa.models.native import ssd_mobilenet


def test_rust_post_processor():
    with open("models/mlcommons_ssd_mobilenet_v1_int8.onnx_truncated.dfg", "rb") as file:
        dfg = file.read();
        processor = ssd_mobilenet.RustPostprocessor(dfg)
