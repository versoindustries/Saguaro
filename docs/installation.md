# Installation Guide

SAGUARO is a hybrid Python/C++ system. It relies on a Native C++ Extension (`saguaro_core.so`) for quantum operations and holographic indexing.

> [!IMPORTANT]
> **Production vs. Development**
> For most users, we recommend the **Development Install** to build the native ops locally for your specific hardware (AVX2/AVX512 optimizations).

## Prerequisites

*   **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS recommended)
*   **Python**: Version 3.12+
*   **Compiler**: C++17 compliant (GCC 11+ or Clang 14+)
*   **Build Tools**: `cmake`, `make`, `ninja-build` (optional but faster)

## Step 1: Clone Repository
```bash
git clone https://github.com/VersoIndustries/Saguaro.git
cd Saguaro
```

## Step 2: System Dependencies (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install cmake build-essential python3.12-dev python3.12-venv
```

## Step 3: Virtual Environment
We strongly recommend a dedicated virtual environment to avoid conflicts with system packages.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## Step 4: Build & Install
There are two ways to install Saguaro:

### Option A: Development Install (Recommended)
This builds the native extension in-place and links the Python package foundation.

```bash
# Install strict dependencies
pip install -r requirements.txt

# Build Native Ops and install in editable mode
pip install -e .
```
*Note: The `setup.py` script automatically invokes CMake to compile `saguaro/native`.*

### Option B: Build Verification
If you encounter issues with the native build, verification scripts are available:

```bash
# Verify Native Ops loading
python3 -c "import saguaro.native; print('Native Ops Loaded Successfully')"
```

## Troubleshooting

### "GLIBCXX_3.4.XX not found"
This usually means your `venv` is picking up an older `libstdc++` or the compiler used for Python differs from your validation compiler.
**Fix**: Ensure `CC` and `CXX` environment variables point to your modern GCC.

```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
pip install -e . --force-reinstall
```

### "No module named 'tensorflow'"
Saguaro uses TensorFlow for graph execution of quantum ops. Ensure it is installed:
```bash
pip install tensorflow
```
