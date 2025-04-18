from setuptools import setup, find_packages

setup(
    name="sillyai",
    version="0.1.0-alpha",
    description="An advanced, lightweight complex-valued transformer model with concept graphing, symbolic reasoning, and real-valued support.",
    author="bumblebee777",
    url="https://github.com/bumbelbee777/sillyai",
    packages=find_packages(),
    install_requires=["torch"],
)