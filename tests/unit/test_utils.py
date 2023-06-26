from furiosa.models.utils import get_nux_version


def test_compiler_version():
    version = get_nux_version()
    assert version
    assert len(version.version) > 0
    assert len(version.revision) > 0
