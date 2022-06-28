import asyncio

from furiosa.models.vision import ResNet18
from furiosa.registry import Model
from furiosa.runtime import session

model: Model = ResNet18()

with session.create(model.model, compile_config=model.compiler_config) as sess:
    # Load input data
    data = ...

    # Pre-process the input data via provided preprocess function by furiosa-artifacts
    input = model.preprocess(data)

    # Run the inference
    output = sess.run(input)

    # Post-process the output data via provided preprocess function by furiosa-artifacts
    final_output = model.postprocess(output)
