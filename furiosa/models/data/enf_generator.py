from multiprocessing import Pool
import os
from pathlib import Path
import tempfile
from typing import Tuple

import onnx
import yaml

import furiosa.quantizer
from furiosa.tools.compiler.api import VersionInfo, compile

QUANTIZER_CONFIG = {"with_quantize": False}
COMPILER_CONFIG = {"tabulate_dequantize": True}
TARGET_NPU = "warboy-2pe"
COMPILED_SUFFIX = "_warboy_2pe"
MAX_WORKER_PROCESSES = 8

base_path = Path(__file__).parent
compiler_version = VersionInfo()
print(f"furiosa-compiler {compiler_version.version} (rev. {compiler_version.git_hash})")
model_directories = [p for p in base_path.glob('*') if p.is_dir() and 'generated' not in p.name]
model_directories = list(enumerate(sorted(model_directories), start=1))
print(f"Found {len(model_directories)} ONNX model files:")
print('\n'.join(map(lambda x: f" {x[0]}. {x[1].name}", model_directories)))
generated_path = base_path / f"generated/{compiler_version.version}_{compiler_version.git_hash}"
generated_path.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {generated_path}")


def set_compiler_config(compiler_config: dict):
    # Make and set compiler config
    fd, compiler_config_path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as f:
        yaml.dump(compiler_config, f)
    os.environ['NPU_COMPILER_CONFIG_PATH'] = compiler_config_path


def quantize_and_compile_model(arg: Tuple[int, Path]):
    index, model_dir_path = arg
    model_full_name = model_dir_path.name
    enf_path = generated_path / f"{model_full_name}{COMPILED_SUFFIX}.enf"
    if enf_path.exists() and enf_path.is_file():
        print(f"  [{index}] {enf_path} already exists, so skipped", flush=True)
        return

    onnx_path = next(model_dir_path.glob('*_f32.onnx'))
    model_short_name = onnx_path.stem.removesuffix('_f32')
    calib_range_path = model_dir_path / f"{model_short_name}.calib_range.yaml"
    print(f"  [{index}] {model_short_name} starts from {onnx_path}", flush=True)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path).SerializeToString()

    # Quantize
    with open(calib_range_path) as f:
        calib_ranges = yaml.full_load(f)
    dfg = furiosa.quantizer.quantize(onnx_model, calib_ranges, **QUANTIZER_CONFIG)
    print(f"  [{index}] {model_short_name} quantized", flush=True)

    compiler_config = dict(COMPILER_CONFIG)
    compiler_config_path = model_dir_path / "compiler_config.yaml"
    if compiler_config_path.exists() and compiler_config_path.is_file():
        print(f"  [{index}] {model_short_name} has extra compiler config: ", flush=True, end='')
        with open(compiler_config_path) as f:
            additional_compiler_config = yaml.full_load(f)
            print(str(additional_compiler_config), flush=True)
            compiler_config.update(additional_compiler_config)
    set_compiler_config(compiler_config)

    # Redirect C lib's stderr to /dev/null
    devnull = open('/dev/null', 'w')
    os.dup2(devnull.fileno(), 2)

    # Compile and write to file
    enf = compile(bytes(dfg), target_npu=TARGET_NPU)
    with open(enf_path, 'wb') as f:
        f.write(enf)
    print(f"  [{index}] {model_short_name} compiled to {enf_path}", flush=True)


if __name__ == '__main__':

    print(f"Spawn {MAX_WORKER_PROCESSES} worker processes")
    print("=" * 80)


    # Do the job parallelly
    with Pool(processes=MAX_WORKER_PROCESSES) as pool:
        pool.map(quantize_and_compile_model, model_directories)
