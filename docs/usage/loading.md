# Loading models

You can load models provided by Furiosa Artifacts using [furiosa-models](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-models) which based on [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry).

## Example

```python
--8<-- "docs/examples/loading_model.py"
```

What's going on here:

`ResNet18(pretrained=True)`

Create model instance. This function ultimately calls the function entrypoint which provided by `artifacts.py` in `furiosa-artifacts`

`pretrained=True` is an arbitrary argument which will transparently pass to model init. You can see what arguments are defined in the model class.

---

`asyncio.run()`

Function entrypoints provided by Furiosa Artifacts are async by default to support concurrency to load models. You need to call the entrypoints in async functions or async eventloop.
