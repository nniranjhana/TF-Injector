# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="TF-Injector",
    version="0.0.1",
    description="A pip package to inject faults into TensorFlow v2 models",
    license="MIT",
    author="Niranjhana Narayanan",
    packages=['src'],
    install_requires=[
        'PyYAML>=5.4.1',
    ],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ]
)
