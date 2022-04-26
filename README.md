Furiosa Artifacts
=================

This repository provides deep learning models provided by FuriosaAI. Available models are described in `artifacts.py` which is a descriptor file in the form defined by [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry/)

## Installation

Load models via [furiosa-models](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-models).

```sh
pip install furiosa-models
```

## Example

```python
import asyncio

from furiosa.registry import Model
from furiosa.models.vision import MLCommonsResNet50


model: Model = asyncio.run(MLCommonsResNet50())
```

## dvc

Model binary files(e.g. `*.onnx`, `*.tflite`) are managed under [DVC](https://dvc.org/).
If you use this repository through high-level abstracted library(e.g. [furiosa-models](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-models) you may skip this part.

### dvc installation
[https://dvc.org/doc/install](https://dvc.org/doc/install)
You probably need `dvc[s3]` package since this repository uses s3 as dvc's backend.

## License

```
Copyright (c) 2021 FuriosaAI Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
