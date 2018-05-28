#!/usr/bin/env python
# coding=utf8

from setuptools import setup, find_packages

version = "0.1.0"

setup(
    name="modelXplore",
    version=version,
    url="https://github.com/celliern/modelXplore.git",
    author="Nicolas Cellier",
    author_email="contact@nicolas-cellier.net",
    description=("Model exploration tool : sensitivity analysis, "
                 "incremental sampler and meta-model tuner which "
                 "provide a surface response for influents parameters"),
    packages=find_packages(),
    install_requires=["SALib",
                      "sklearn",
                      "optunity",
                      "voluptuous",
                      "fuzzywuzzy",
                      "python-Levenshtein",
                      "matplotlib",
                      "xarray",
                      "pendulum"],
)
