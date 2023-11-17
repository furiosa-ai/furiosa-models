from itertools import product
from multiprocessing import Pool
import os
from pathlib import Path
import re
import shlex
import subprocess as sp
import tempfile
from typing import Tuple

import onnx
import yaml

from furiosa.quantizer import ModelEditor, TensorType, get_pure_input_names, quantize

COMPILER_CONFIG = {"lower_tabulated_dequantize": True}

compiler_info = sp.run(
    ['furiosa-compiler', '--version'], capture_output=True, text=True, check=True
).stdout
pattern = r"backend:\n- version: (.+?)\n- revision: (.+?)\b"
match = re.search(pattern, compiler_info)
if match:
    version = match.group(1)
    git_hash = match.group(2)
else:
    raise RuntimeError("Cannot parse furiosa-compiler version info")

base_path = Path(__file__).parent
print(f"furiosa-compiler {version} (rev. {git_hash})")
model_directories = [p for p in base_path.glob('*') if p.is_dir() and 'generated' not in p.name]
model_directories = list(enumerate(sorted(model_directories), start=1))
print(f"Found {len(model_directories)} ONNX model files:")
print('\n'.join(map(lambda x: f" {x[0]}. {x[1].name}", model_directories)))
generated_path = base_path / f"generated/{version}_{git_hash}"
generated_path.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {generated_path}")


def quantize_and_compile_model(arg: Tuple[int, Path, int]):
    index, model_dir_path, num_pe = arg
    model_full_name = model_dir_path.name
    enf_path = generated_path / f"{model_full_name}_warboy_{num_pe}pe.enf"
    if enf_path.exists() and enf_path.is_file():
        print(f"  [{index}] {enf_path} already exists, so skipped", flush=True)
        return

    onnx_path = next(model_dir_path.glob('*_f32.onnx'))
    model_short_name = onnx_path.stem.removesuffix('_f32')
    calib_range_path = model_dir_path / f"{model_short_name}.calib_range.yaml"
    print(f"  [{index}] {model_short_name} starts from {onnx_path}", flush=True)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Quantize
    with open(calib_range_path) as f:
        calib_ranges = yaml.full_load(f)
    editor = ModelEditor(onnx_model)
    for input_name in get_pure_input_names(onnx_model):
        editor.convert_input_type(input_name, TensorType.UINT8)
    quantized_onnx = quantize(onnx_model, calib_ranges)
    with tempfile.NamedTemporaryFile() as quantized_onnx_file:
        onnx.save(quantized_onnx, quantized_onnx_file.name)
        print(f"  [{index}] {model_short_name} quantized", flush=True)

        # Acquire compiler config
        compiler_config = dict(COMPILER_CONFIG)
        additional_config = model_dir_path / "compiler_config.yaml"
        if additional_config.exists() and additional_config.is_file():
            print(f"   [{index}] {model_short_name} has extra compiler config: ", flush=True)
            with open(additional_config) as f:
                additional_compiler_config = yaml.full_load(f)
                print(str(additional_compiler_config), flush=True)
                compiler_config.update(additional_compiler_config)

        # Write compiler config to temp file
        fd, compiler_config_path = tempfile.mkstemp()
        with os.fdopen(fd, "w") as f:
            yaml.dump(compiler_config, f)

        # Compile and write to file
        target_npu = "warboy" if num_pe == 1 else "warboy-2pe"
        cmd = (
            f'furiosa-compiler --target-npu {target_npu} --output {enf_path} '
            f'{quantized_onnx_file.name}'
        )
        sp.run(
            shlex.split(cmd),
            env={"NPU_COMPILER_CONFIG_PATH": compiler_config_path},
            check=True,
            stderr=sp.DEVNULL,
        )

        # Manually remove compiler config file (as we use mkstemp)
        Path(compiler_config_path).unlink()

        print(f"  [{index}] {model_short_name} compiled to {enf_path}", flush=True)


if __name__ == '__main__':
    lst = product(model_directories, [1, 2])  # [(idx, model_dir_path, num_pe), ...]
    lst = [(a, b, c) for (a, b), c in lst]  # Flatten list

    # Do the job parallelly
    with Pool() as pool:
        pool.map(quantize_and_compile_model, lst)
