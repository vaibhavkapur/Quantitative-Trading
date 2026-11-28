from setuptools import find_packages, setup

setup(
    name="multi-factor-asset-pricing-engine",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "mfape=mfape.cli:main",
        ]
    },
)
