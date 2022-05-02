import asyncio

from furiosa.models.vision import ResNet18
from furiosa.registry import Model
from furiosa.runtime import session

model: Model = asyncio.run(ResNet18())

with session.create(model.model) as session:
    # Load input data
    data = ...

    # Pre-process the input data via provided preprocess function by furiosa-artifacts
    input = model.preprocess(data)

    output = session.run(input)

    # Post-process the output data via provided preprocess function by furiosa-artifacts
    model.postprocess(output)
