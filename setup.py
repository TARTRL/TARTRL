#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import os
from setuptools import setup
import setuptools


def get_install_requires() -> list:
    return ["setuptools>=50.0", "treevalue", "numpy"]


def get_extra_requires() -> dict:
    req = {
        "test": [
            "pytest",
            "pytest-cov",
        ],
    }
    return req


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("tartrl", "__init__.py"), "r").read().split()
    return init[init.index("__VERSION__") + 2][1:-1]


setup(
    name="tartrl",
    version=get_version(),
    description="reinforcement learning framework",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="tartrl contributors",
    author_email="contact@tartrl.cn",
    url="https://github.com/TARTRL/TARTRL",
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["tartrl=tartrl.cli.cli:run"]},
    project_urls={
        "Code": "https://github.com/TARTRL/TARTRL",
        "Documentation": "https://tartrl-docs.readthedocs.io/zh/latest/",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords="reinforcement-learning multi-agent "
    "reinforcement-learning-algorithms pytorch machine-learning "
    "baselines toolbox python data-science gym gymnasium",
    python_requires=">=3.8",
    install_requires=get_install_requires(),
    extras_require=get_extra_requires(),
)
