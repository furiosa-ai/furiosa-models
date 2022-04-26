Furiosa Artifacts
=================

This repository provides deep learning models provided by FuriosaAI. Available models are described in `artifacts.py` which is a descriptor file in the form defined by [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry/)

## Available models

### Image Classification

Model | Python class name | Pretrained? |
:------------ | :-------------|:-------------:|
| ResNet50-v1.5 | `MLCommonsResNet50` | :heavy_check_mark: |
| EfficientNetV2_S | `EfficientNetV2_S` | :heavy_check_mark: |

### Object detection

Model | Python class name | Pretrained? |
:------------ | :-------------|:-------------:|
| SSD-ResNet34 | `MLCommonsSSDResNet34` | :heavy_check_mark: |
| SSD-MobileNets-v1 | `MLCommonsSSDMobilenet` | :heavy_check_mark: |

See `./artifacts.py` for more detail.

## How to use models in this repository?

See [client side code](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry/#getting-started) in furiosa-reigstry.


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
