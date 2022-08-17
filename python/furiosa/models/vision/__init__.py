from inspect import iscoroutinefunction

from furiosa.common.thread import synchronous
from furiosa.registry import Model

# Where nonblocking models reside
from . import nonblocking

__all__ = []

# Iterate over non-blocking versions of Model classes (that of .nonblocking.vision)
for model in [
    getattr(nonblocking, m)
    for m in dir(nonblocking)
    if iscoroutinefunction(getattr(nonblocking, m))
]:
    # Export synchronous version of Model class in this module scope
    name = model.__name__
    if name[0].isupper():
        globals()[name] = synchronous(model)
        __all__.append(name)

# Clean up unnecessary variables in this module
del iscoroutinefunction, Model, model, name, synchronous
