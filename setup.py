from setuptools import find_namespace_packages, setup
from setuptools_rust import Binding, RustExtension

setup(
    packages=find_namespace_packages(where="python"),
    package_dir={"": "python"},
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
)
