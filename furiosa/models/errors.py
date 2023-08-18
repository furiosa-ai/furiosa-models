class FuriosaModelException(Exception):
    """General exception caused by Furiosa Runtime"""

    def __init__(self, message: str):
        self._message = message

    def __repr__(self):
        return '{}'.format(self._message)

    def __str__(self):
        return self.__repr__()


class ArtifactNotFound(FuriosaModelException):
    """Certain artifact file not found"""

    def __init__(self, artifact_file_name: str):
        super().__init__(f"{artifact_file_name} is required, but missing")


class NotFoundInDVCRemote(ArtifactNotFound):
    """Certain artifact file's DVC exists, but cannot fetch from remote"""

    def __init__(self, artifact_file_name: str, md5sum: str):
        super().__init__(f"{artifact_file_name} not exists in DVC remote\nmd5sum: {md5sum}")


class VersionInfoNotFound(FuriosaModelException):
    """Could not retrieve compiler version information"""

    def __init__(self):
        super().__init__(
            "Could not retrieve furiosa compiler information. Try: `pip install furiosa-sdk`"
        )


class ExtraPackageRequired(FuriosaModelException):
    """Needs extra packges to quantize and compile"""

    def __init__(self):
        super().__init__(
            "Needs extra packges to quantize and compile manually. Try: `pip install "
            "furiosa-models[full]`"
        )
