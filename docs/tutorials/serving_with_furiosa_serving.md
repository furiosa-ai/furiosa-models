# Serving Example with Furiosa Serving

Furiosa Serving is a lightweight library based on [FastAPI](https://fastapi.tiangolo.com/) that allows you to run a model server on a Furiosa NPU.

For more information about Furiosa Serving, you can visit the [package link](https://pypi.org/project/furiosa-serving/).

## Getting Started

To get started with Furiosa Serving, you'll need to install the [furiosa-serving library](https://pypi.org/project/furiosa-serving/), create a ServeAPI (which is a `FastAPI` wrapper), and set up your model for serving.
In this example, we'll walk you through the steps to create a simple ResNet50 server.

First, you'll need to import the necessary modules and initialize a FastAPI app:

```python
from tempfile import NamedTemporaryFile
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn

from furiosa.common.thread import synchronous
from furiosa.models import vision
from furiosa.serving import ServeAPI, ServeModel

serve = ServeAPI()
app: FastAPI = serve.app
```

## Model Initialization

Next, you can initialize a vision model, such as ResNet50, for serving:

```python
resnet50 = vision.ResNet50()

model_file = NamedTemporaryFile()
model_file.write(resnet50.model_source())
model_file_path = model_file.name
model: ServeModel = synchronous(serve.model("furiosart"))(
    'ResNet50', location=model_file_path
)
```

!!!note
    ServeModel does not support in-memory model binaries for now. Instead, you can write the model into a temporary file and pass its path like example.

## Model Inference

Now that you have your FastAPI app and model set up, you can define an endpoint for model inference. In this example, we create an endpoint that accepts an image file and performs inference using ResNet50:

```python
@model.post("/infer")
async def infer(image: UploadFile = File(...)) -> Dict[str, str]:
    # Model Zoo's preprocesses do not consider in-memory image file for now,
    # so we write in-memory image into a temporary file and pass its path
    image_file_path = NamedTemporaryFile()
    image_file_path.write(await image.read())

    tensors, _ctx = resnet50.preprocess(image_file_path.name)

    # Infer from ServeModel
    result: List[np.ndarray] = await model.predict(tensors)

    response: str = resnet50.postprocess(result)

    return {"result": response}
```

## Running the Server

Finally, you can run the FastAPI server using [uvicorn](https://www.uvicorn.org/).

```python
# Run the server if the current Python script is called directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Alternatively, you can run [uvicorn](https://www.uvicorn.org/) server via internal app variable from ServeAPI instance like [normal FastAPI application](https://fastapi.tiangolo.com/tutorial/first-steps/#first-steps).

```shell
$ uvicorn main:app # or uvicorn main:serve.app
```

This example demonstrates the basic setup of a FastAPI server with Furiosa Serving for model inference. You can extend this example to add more functionality to your server as needed.

For more information and advanced usage of Furiosa Serving, please refer to the [Furiosa Serving documentation](https://pypi.org/project/furiosa-serving/).


You can find the full code example here.

```python
--8<-- "docs/examples/serving.py"
```