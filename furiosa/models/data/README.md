# Model Artifacts

This directory includes the model artifacts which are managed by DVC.

## DFG/ENF Generator
### Prerequisites

Please install necessary packages according to [Prerequisites](https://github.com/furiosa-ai/furiosa-models/wiki#prerequisites).
Then, install the python dependencies as following:
```
pip install -r furiosa-models/models/requirements.txt
```

### Compiler Config
You can optionally add a compiler config for each model or model's specific IR formats.

If you want to set a compiler config for a `mlcommons_resnet50_v1.5_int8.onnx`,
you need to add `mlcommons_resnet50_v1.5_int8.yaml` here.

`enf-generator.sh` also allow to specify a compiler config for a certain IR format.
To specify a compiler config for dfg or enf, you can add the config file named `{NAME}.{FORMAT}.yaml`; e.g.,
* DFG: `mlcommons_resnet50_v1.5_int8.dfg.yaml`
* ENF: `mlcommons_resnet50_v1.5_int8.enf.yaml`

### Run
```sh
$ enf_generator.sh

./enf_generator.sh
[+] Detected version of compiler: 0.8.0-dev (rev. bd3a389dc)
[+] Installed version of compiler package: 0.8.0-2+nightly-220827
[+] Found 8 ONNX Files:
 [1] mlcommons_resnet50_v1.5_int8.onnx
 [2] mlcommons_resnet50_v1.5_int8_truncated.onnx
 [3] mlcommons_ssd_mobilenet_v1_int8.onnx
 [4] mlcommons_ssd_mobilenet_v1_int8_truncated.onnx
 [5] mlcommons_ssd_resnet34_int8.onnx
 [6] mlcommons_ssd_resnet34_int8_truncated.onnx
 [7] yolov5l_int8.onnx
 [8] yolov5m_int8.onnx
[+] Output directory: /home/hyunsik/Code/furiosa-models-2/models/generated/0.8.0-dev_bd3a389dc
[1/8] Compiling mlcommons_resnet50_v1.5_int8.onnx ..
 [Task 1/2] Generating mlcommons_resnet50_v1.5_int8_warboy_2pe.dfg ... (Skipped)
 [Task 2/2] Generating mlcommons_resnet50_v1.5_int8_warboy_2pe.enf ... (Skipped)
...
```
