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

    def __init__(self, model_name: str, ir: str):
        super().__init__(f"'{ir} for {model_name} is required, but missing'")


class VersionInfoNotFound(FuriosaModelException):
    """Could not retrieve compiler version information"""

    def __init__(self):
        super().__init__(
            f"Could not retrieve furiosa compiler information. Try: `pip install furiosa-sdk`."
        )
