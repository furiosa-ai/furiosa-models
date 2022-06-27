# Running models

You can run inference via models provided by Furiosa Artifacts using [furiosa-runtime](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-runtime).

## Example

```python
--8<-- "docs/examples/running_model.py"
```

`with session.create(model.model, compile_config=model.compiler_config) as sess:`

Session in `furiosa-runtime` needs model binary when creating the session. As models provided by `furiosa-models` have `model` field which is bytes formatted model binary, you can pass the model into the session. Furiosa Artifacts also provides recommended compiler configurations for some models. You can pass the `model.compiler_config` variable when creating a session. Of course you can pass your own compiler configuration as well.

---

`input = model.preprocess(data)`

`final_output = model.postprocess(output)`

Model pre/post processing via the functions provided by Furiosa Artifacts. There may be other functions in `Model` class provided by model providers. As a model client, you should find the documents to find which functions are available.

---

`output = sess.run(input)`

Run inference via the session.

## More references

You can find Furiosa runtime API references and examples in [SDK documentation](https://furiosa-ai.github.io/docs/latest/en/#furiosaai-sdk-tutorial-and-examples)
