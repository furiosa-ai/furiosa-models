# Loading models

You can load models provided by Furiosa Artifacts using [furiosa-models](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-models) which based on [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry).

## Example

### Blocking and Non-Blocking API
Furiosa Model offers both blocking and non-blocking API for fetching models.

#### Blocking example
```python
--8<-- "docs/examples/loading_model.py"
```

#### Non-blocking example
```python
--8<-- "docs/examples/loading_model_nonblocking.py"
```

What's going on here:

`ResNet18(pretrained=True)`

Create model instance. This function ultimately calls the function entrypoint which provided by `artifacts.py` in `furiosa-artifacts`

If `pretrained=True` is set, a model with pre-trained weights will be fetched. `pretrained=True` is a default option as well as the only-allowed option for now. You can also find more arguments in the model class.

---

For non-blocking example

`asyncio.run()`

`furiosa.models.nonblocking` module offers non-blocking API. When you are writing codes using furiosa-models in async functions or async eventloop, you should use the non-blocking APIs.
