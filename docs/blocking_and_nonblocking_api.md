# Blocking and Non-Blocking API
Furiosa Model offers both blocking and non-blocking API for fetching models. 
Since the sizes of model images vary from tens to hundreds megabytes,  
fetching a single model takes some time (usually few seconds). If you fetch models within asynchronous code, 
you need to handle this kind of blocking calls. In the case, you can use Furiosa Model's non-blocking API.

## Blocking example
```python
--8<-- "docs/examples/loading_model.py"
```

## Non-blocking example
To use non-blocking API, you just need to import `furiosa.models.vision.nonblocking` instead of `furiosa.models.vision`.
Creating your model should be executed within `asyncio.run()` block or `async` function with `await` keyword.
All other API of models is the same as blocking API.

```python
--8<-- "docs/examples/loading_model_nonblocking.py"
```