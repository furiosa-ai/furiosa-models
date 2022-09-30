from furiosa.models.utils import compiler_version


def test_compiler_version():
    version = compiler_version()
    assert version
    assert len(version.version) > 0
    assert len(version.revision) > 0
