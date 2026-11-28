"""Compatibility shim for editable installs on older pip/setuptools versions."""

from setuptools import find_packages, setup

setup(
    name="optimal-execution-research-engine",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "optimal-execution=optimal_execution.cli:main",
        ],
    },
)
