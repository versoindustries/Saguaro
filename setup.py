from setuptools import setup, find_packages

setup(
    name="saguaro-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
        "ruff",
        "mypy",
        "vulture",
    ],
    entry_points={
        "console_scripts": [
            "saguaro=saguaro.cli:main",
        ],
    },
    include_package_data=True,
    description="Quantum Codebase Operating System (Q-COS)",
    author="Verso Industries",
    python_requires=">=3.8",
)
