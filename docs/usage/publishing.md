# Publishing models

Furiosa Artifacts use model class in [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry/) to publish models. `furiosa-registry` will find available models via a descriptor file named `artifacts.py`.

To publish models, as a model provider you need to add function entrypoints which returns a model instance in `artifacts.py`

## Example

``` python title="artifacts.py"
--8<-- "docs/examples/publish_model.py"
```

What's going on here:

`class ResNet18_Model(Model)`

Model class implements `furiosa.registry.Model`. `furiosa.registry.Model` is a class which you have to implement to publish your model. This model class is based on a [pydantic Model](https://pydantic-docs.helpmanual.io/usage/models/) and have several required fields to fill including:

- `name` model name.
- `model` model binary bytes.
- `format` model binary format. "onnx" or "tflite" are supported.

---

`def preprocess(self, *args: Any, **kwargs: Any) -> Any`

`def postprocess(self, *args: Any, **kwargs: Any) -> Any`

Additional functions to suppport model modification. As a model provider, you can add pre-process, post-process functions to provide model specific functionalities. Note that this custom functions are not defined via interface which means you can add any custom functions.

As users does not know which functions are provided, you have to document these functions to allow clients to use models correctly.

---

`async def ResNet18(*args: Any, **kwargs: Any) -> ResNet18_Model`

Function entrypoint which will be called from `furiosa-registry`.

---

`model=await loader.read("models/resnet18.onnx")`

Loading model binary bytes via `furiosa.registry.client.transport`. You may use different ways to load binary bytes depending on how you maintain your model binary.
