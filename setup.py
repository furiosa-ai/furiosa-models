from distutils.command.build_ext import build_ext as build_ext_orig

from setuptools import Extension, find_namespace_packages, setup
from setuptools_rust import Binding, RustExtension


class SharedObjectCTypesExtension(Extension):
    pass


class so_build_ext(build_ext_orig):
    def __init__(self, *args, **kwargs):
        self._ctypes = False
        super().__init__(*args, **kwargs)

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, SharedObjectCTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            from distutils.sysconfig import get_config_var

            ext_suffix = get_config_var('EXT_SUFFIX').split(".")
            return ext_name + f".{ext_suffix[-1]}"
        return super().get_ext_filename(ext_name)


setup(
    packages=find_namespace_packages(where="python"),
    package_dir={
        "": "python",
    },
    ext_modules=[
        SharedObjectCTypesExtension(
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
    cmdclass={"build_ext": so_build_ext},
)
