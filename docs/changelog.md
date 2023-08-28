# Changelog

## [0.10.0 - 2023-08-28]

### New Features
- Provide 1pe artifacts too #158

### Improvements
- Resolve artifact binaries lazily #155
- Added `ruff` linter #160
- Automatically build documentation on comment #162
- Upgrade `pydantic` library version to `2.0.0` #166
- Added `model.resolve_all()` to resolve all lazily loaded fields at once #166
- Added local artifact binary cache #166

## Removed
- Removed unused `timm` dependency #149
- **Breaking:** Now uses default Python initializer instead of `model.load()` #166
- **Breaking:** `model.enf` field has been removed #166
- **Breaking:** `model.source`, `model.calib_yaml` fields have been renamed #166

## [0.9.1 - 2023-05-26]

## [0.9.0 - 2023-05-12]

### New Features
- Add EfficientNetB0 model #121
- Add EfficientNetV2-S model #130
- Set default target as Warboy's production revision (B0) #125
- Provide calibration ranges for every model #144

### Improvements
- Removed `Quantize` external operators #144
- Detailed error messages for model file fetching #144
- ENF generator can do the jobs parallelly #144
- Removed furiosa.registry dependency #144
- Faster import for furiosa.models #117
- Replace yolov5's box decode implementation in Rust #109
- Remove Cpp postprocessor implementations #102
- Change packaging tool from setuptools-rust to flit #109

## Removed
- Truncated models and corresponding postprocesses #144
- **Breaking:** drop support of directly passing Model to session.create() #144

### Tasks
- Release guide for developers #129
- Report regression test's result with PR comment #110

### Bug Fixes
- Fix CLI to properly report net inference time #112
- Update certain model sources with valid onnx #120

## [0.8.0 - 2022-11-10]

### New Features
- Add ResNet50 model
- Add SSD ResNet34 model
- Add SSD MobileNet model
- Add YOLOv5l model
- Add YOLOv5m model

### Improvements
- Add native postprocessing implementation for ResNet50 #42
- Add native postprocessing implementation for SSD ResNet34 #45
- Add native postprocessing implementation for SSD MobileNet #16
- Refactor Model API to use classmethods to load models #66
- Make Model Zoo's ABC Model #78
- Retain only necessary python dependencies #81
- Improve enf_generator.sh and add enf, dfg models for e2e tests #33
- Add mkdocstrings to make API references #22

### Tasks
- Regresstion Test #61
- Attach Tekton CI #41
- Yolov5 L/M e2e testing code #22
- Change the documentation layout and add resnet34, resnet50, mobilenet details #33
- Replace maturin with setuptools-rust #26

### Bug Fixes
- Resolve DVC directly #80
