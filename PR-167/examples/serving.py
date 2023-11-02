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

resnet50 = vision.ResNet50()
# ServeModel does not support in-memory model binary for now,
# so we write model into temp file and pass its path
model_file = NamedTemporaryFile()
model_file.write(resnet50.model_source())
model_file_path = model_file.name

model: ServeModel = synchronous(serve.model("furiosart"))(
    'ResNet50', location=model_file_path
)


@model.post("/infer")
async def infer(image: UploadFile = File(...)) -> Dict[str, str]:
    # Model Zoo's preprocesses do not consider in-memory image file for now
    # (note that it's different from in-memory tensor)
    # so we write in-memory image into temp file and pass its path
    image_file_path = NamedTemporaryFile()
    image_file_path.write(await image.read())

    tensors, _ctx = resnet50.preprocess(image_file_path.name)

    # Infer from ServeModel
    result: List[np.ndarray] = await model.predict(tensors)

    response: str = resnet50.postprocess(result)

    return {"result": response}


# Run the server if current Python script is called directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
