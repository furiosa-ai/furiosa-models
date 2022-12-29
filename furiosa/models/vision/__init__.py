from typing import Any, List

__all__ = ["ResNet50", "SSDMobileNet", "SSDResNet34", "YOLOv5l", "YOLOv5m"]

_class_modules = {
    "ResNet50": ".resnet50",
    "SSDMobileNet": ".ssd_mobilenet",
    "SSDResNet34": ".ssd_resnet34",
    "YOLOv5l": ".yolov5",
    "YOLOv5m": ".yolov5",
}


def __getattr__(name: str) -> Any:
    import importlib

    module = importlib.import_module(_class_modules.get(name, "." + name), __name__)
    cls = getattr(module, name)
    globals()[name] = cls  # so that __getattr__ won't be called again
    return cls


def __dir__() -> List[str]:
    return __all__
