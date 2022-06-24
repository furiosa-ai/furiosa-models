# Model inference testing scripts

Use these scripts to test Model before attaching NPU to the CI.

## Tested Models
### Image Classification
- [x] ResNet50
### Object Detection
- [x] ResNet34
- [x] MobileNet

## How to run
```shell
pip install --upgrade pip
pip install .[test]
pip install furiosa-runtime
```

To run the test,`pytest ./scripts` in base directory of this repository.
