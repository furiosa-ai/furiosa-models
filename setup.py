from setuptools import Extension, find_namespace_packages, setup
from setuptools_rust import Binding, RustExtension

setup(
    packages=find_namespace_packages(where="python"),
    package_dir={
        "": "python",
    },
    ext_modules=[
        Extension(
            "furiosa.models.vision.yolov5.box_decode.cbox_decode.box_decode.cbox_decode",
            [
                "python/furiosa/models/vision/yolov5/box_decode/cbox_decode/box_decode/box_decode.cpp"
            ],
            extra_compile_args=["-ffast-math", "-O3"],
            language="c++",
        ),
    ],
    rust_extensions=[
        RustExtension(
            "furiosa.models.furiosa_models_vision_native",
            path="Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
    install_requires=["numpy"],
)
