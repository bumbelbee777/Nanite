from setuptools import find_packages, setup, Extension
import pybind11

ext_modules = [
    Extension(
        "nanite.backend.BitplaneEngine",
        ["nanite/backend/BitplaneEngine.cxx"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/O2", "/std:c++20"],
    )
]

setup(
    name="nanite",
    version="0.1.0",
    description="An advanced, lightweight complex-valued neuro-symbolic transformer model.",
    author="bumblebee777",
    url="https://github.com/bumbelbee777/nanite",
    packages=find_packages(),
    install_requires=["torch", "numpy", "numba", "imageio", "pytest", "beautifulsoup4"],
    ext_modules=ext_modules,
)
