from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="hhh",
    author="Javier Duarte",
    url="https://github.com/ucsd-hep-ex/hhh",
    license="MIT",
    install_requires=[
        "coffea",
        "spanet",
        "numpy",
    ],
)
