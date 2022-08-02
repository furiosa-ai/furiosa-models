from distutils.command.build_ext import build_ext as build_ext_orig

from setuptools import Extension, find_packages, setup


class CTypesExtension(Extension):
    pass


class build_ext(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


setup(
    name="cbox_decode",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[
        CTypesExtension(
            "box_decode/cbox_decode",
            ["box_decode/box_decode.cpp"],
            extra_compile_args=["-ffast-math", "-O3"],
        ),
    ],
    install_requires=["numpy"],
    cmdclass={"build_ext": build_ext},
)
