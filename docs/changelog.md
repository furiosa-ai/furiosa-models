# Changelog

## [unreleased]
### Added

### Changed
- Changed packaging tool from setuptools-rust to hatch #102

### Deprecated

### Removed
- MLPerf postprocess rust bindings #102
- Cpp postprocessor(s) #102

### Fixed

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