# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from setuptools import setup, find_packages

setup(
    name="pointbridge",
    packages=[
        package for package in find_packages() if package.startswith("pointbridge")
    ],
    install_requires=[],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3.11',
    description="Point Bridge: 3D Representations for Cross Domain Policy Learning",
    author="Siddhant Haldar",
    url="https://github.com/NVlabs/pointbridge",
    author_email="siddhanthaldar@nyu.edu",
    version="1.0.0",
    long_description="This the official code for the paper 'Point Bridge: 3D Representations for Cross Domain Policy Learning'.",
    long_description_content_type='text/markdown'
)