from furiosa.models.utils import compiler_version, generated_artifact_path


def test_compiler_version():
    version = compiler_version()
    assert version
    assert len(version.version) > 0
    assert len(version.revision) > 0


def test_generated_artifact_path():
    path = generated_artifact_path("models/mlcommons_ssd_resnet34_int8.onnx_truncated.onnx", "enf")
    print(path)