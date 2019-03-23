#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="candels-blender",
    version="0.1.0",
    description="Create galaxy blends from CANDELS galaxies images.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aboucaud/candels-blender",
    author="Alexandre Boucaud",
    author_email="aboucaud@apc.in2p3.fr",
    packages=find_packages(),
    license="BSD",
    classifiers=[
        "Intended Audience :: Science/Research"
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires = [
        "numpy",
        "scipy",
        "astropy",
        "pandas",
        "click",
    ],
    python_requires='>=3.6',
    entry_points = {
        "console_scripts": [
            "candels-blender = blender.scripts.cli:cli",
        ],
    },
    zip_safe=False,
    include_package_data=True,
)
