# Running models

You can run inference via models provided by Furiosa Artifacts using [furiosa-runtime](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-runtime).

## Example

```python
--8<-- "docs/examples/running_model.py"
```

`with session.create(model.model) as session:`

Session in `furiosa-runtime` needs model binary when creating the session. As models provided by `furiosa-models` have `model` field which is bytes formatted model binary, you can pass the model into the session.

---

`input = model.preprocess(data) `

`output = model.postprocess(output) `

Model pre/post processing via the functions provided by Furiosa Artifacts. There may be other functions in `Model` class provided by model providers. As a model client, you should find the documents to find which functions are available.

---

`output = session.run(input)`

Run inference via the session.

## More references

You can find Furiosa runtime API references and examples in [SDK documentation](https://furiosa-ai.github.io/docs/latest/en/#furiosaai-sdk-tutorial-and-examples)
