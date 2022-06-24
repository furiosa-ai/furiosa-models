from typing import Any

from furiosa.registry import Format, Metadata, Model, Publication
from furiosa.registry.client.transport import FileTransport

loader = FileTransport()


class ResNet18_Model(Model):
    """# This docstring shows up in furisao.registry.help()
    ResNet 18 model

    # Additional arguments
    pretrained (bool): kwargs, load pretrained weights into th model
    """

    def __init__(self, pretrained: bool):
        ...

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        ...


async def ResNet18(*args: Any, **kwargs: Any) -> ResNet18_Model:
    return ResNet18_Model(
        name="ResNet18",
        model=await loader.read("models/resnet18.onnx"),
        format=Format.ONNX,
        family="ResNet",
        version="v1.0",
        metadata=Metadata(
            description="Model description",
            publication=Publication(url="Model publication URL"),
        ),
        **kwargs,
    )
