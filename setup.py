"""
SAGUARO setup.py — Backward Compatibility Shim

Primary packaging configuration is in pyproject.toml.
This file exists for `python setup.py develop` and legacy workflows.
"""
from setuptools import setup, find_packages
import re


def _get_version():
    """Read version from saguaro/__init__.py to keep single source of truth."""
    with open("saguaro/__init__.py", "r") as f:
        match = re.search(r'__version__\s*=\s*"([^"]+)"', f.read())
        return match.group(1) if match else "0.0.0"


setup(
    name="saguaro-core",
    version=_get_version(),
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pyyaml>=6.0",
        "ruff>=0.4",
    ],
    extras_require={
        "tf": ["tensorflow>=2.15"],
        "dev": ["pytest>=7.0", "mypy", "vulture"],
        "enterprise": [
            "tensorflow>=2.15",
            "pytest>=7.0",
            "mypy",
            "vulture",
            "psutil",
        ],
    },
    entry_points={
        "console_scripts": [
            "saguaro=saguaro.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "saguaro": ["*.so", "*.dylib", "data/**/*", "artifacts/**/*"],
    },
    description="Quantum Codebase Operating System (Q-COS)",
    author="Verso Industries",
    python_requires=">=3.10",
)

