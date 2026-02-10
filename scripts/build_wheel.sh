#!/usr/bin/env bash
# =============================================================================
# SAGUARO Build & Package Automation
# Enterprise-grade wheel builder with native C++ extension support
# =============================================================================
#
# Usage:
#   ./scripts/build_wheel.sh              # Build + package wheel
#   ./scripts/build_wheel.sh --no-cmake   # Package only (pre-built .so files)
#   ./scripts/build_wheel.sh --clean      # Clean build from scratch
#
# Requirements:
#   - CMake >= 3.18
#   - C++17 compatible compiler (GCC 9+ or Clang 10+)
#   - TensorFlow installed in the active Python environment
#   - pip, wheel, build packages
#
# Output:
#   - dist/saguaro_core-*.whl  (platform-specific wheel)
#   - saguaro/_saguaro_core.so  (also copied to source tree for dev)
#   - saguaro/_saguaro_native.so
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# =============================================================================
# Parse Arguments
# =============================================================================
NO_CMAKE=false
CLEAN=false

for arg in "$@"; do
    case $arg in
        --no-cmake) NO_CMAKE=true ;;
        --clean)    CLEAN=true ;;
        --help|-h)
            echo "Usage: $0 [--no-cmake] [--clean] [--help]"
            echo ""
            echo "Options:"
            echo "  --no-cmake   Skip CMake build (use pre-built .so files)"
            echo "  --clean      Clean build from scratch"
            echo "  --help       Show this help"
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Checks
# =============================================================================
info "Checking build environment..."

# Python
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    error "Python not found. Set PYTHON env var or ensure python3 is on PATH."
    exit 1
fi
PYTHON_VERSION=$("$PYTHON" --version 2>&1)
ok "Python: $PYTHON_VERSION"

# pip / build
if ! "$PYTHON" -m pip --version &>/dev/null; then
    error "pip not available. Install with: $PYTHON -m ensurepip"
    exit 1
fi

# CMake (optional if --no-cmake)
if [ "$NO_CMAKE" = false ]; then
    if ! command -v cmake &>/dev/null; then
        error "CMake not found. Install with: apt install cmake (or use --no-cmake)"
        exit 1
    fi
    CMAKE_VERSION=$(cmake --version | head -1)
    ok "CMake: $CMAKE_VERSION"

    # C++ compiler
    CXX="${CXX:-$(command -v g++ || command -v clang++ || echo '')}"
    if [ -z "$CXX" ]; then
        error "No C++ compiler found. Install GCC or Clang."
        exit 1
    fi
    ok "C++ Compiler: $CXX"
fi

# =============================================================================
# Clean (optional)
# =============================================================================
if [ "$CLEAN" = true ]; then
    info "Cleaning previous build artifacts..."
    rm -rf "${BUILD_DIR}"
    rm -rf "${PROJECT_ROOT}/dist"
    rm -rf "${PROJECT_ROOT}"/*.egg-info
    rm -f "${PROJECT_ROOT}/saguaro/_saguaro_core.so"
    rm -f "${PROJECT_ROOT}/saguaro/_saguaro_native.so"
    ok "Clean complete."
fi

# =============================================================================
# Step 1: Build Native Libraries (CMake)
# =============================================================================
if [ "$NO_CMAKE" = false ]; then
    info "Building native C++ extensions..."

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # Configure
    cmake "${PROJECT_ROOT}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        2>&1 | tail -5

    # Build
    NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build . --parallel "$NPROC" 2>&1

    ok "Native libraries built successfully."

    # Install (copies .so to saguaro/ source tree)
    cmake --install . 2>&1 | tail -3
    ok "Libraries installed to source tree."

    cd "${PROJECT_ROOT}"
else
    info "Skipping CMake build (--no-cmake)."
fi

# =============================================================================
# Step 2: Verify .so files exist
# =============================================================================
info "Verifying native libraries..."

SO_FOUND=0
for lib in _saguaro_core.so _saguaro_native.so; do
    if [ -f "${PROJECT_ROOT}/saguaro/${lib}" ]; then
        SIZE=$(du -h "${PROJECT_ROOT}/saguaro/${lib}" | cut -f1)
        ok "Found saguaro/${lib} (${SIZE})"
        SO_FOUND=$((SO_FOUND + 1))
    else
        warn "Missing saguaro/${lib}"
    fi
done

if [ "$SO_FOUND" -eq 0 ]; then
    error "No .so files found! Build with CMake first or place .so files in saguaro/"
    exit 1
fi

# =============================================================================
# Step 3: Build Python Wheel
# =============================================================================
info "Building Python wheel..."

# Ensure build tools are available
"$PYTHON" -m pip install --quiet --upgrade pip wheel build 2>/dev/null

# Build the wheel (uses pyproject.toml)
cd "${PROJECT_ROOT}"
"$PYTHON" -m build --wheel --outdir dist/ 2>&1

ok "Wheel built successfully!"

# =============================================================================
# Step 4: Summary
# =============================================================================
echo ""
echo "============================================="
echo "  SAGUARO Build Complete"
echo "============================================="
echo ""

if [ -d "${PROJECT_ROOT}/dist" ]; then
    info "Wheel packages:"
    ls -lh "${PROJECT_ROOT}/dist/"*.whl 2>/dev/null || warn "No wheel files found"
fi

echo ""
info "Install with:"
echo "  pip install dist/saguaro_core-*.whl"
echo ""
info "Install with TensorFlow support:"
echo "  pip install 'dist/saguaro_core-*.whl[tf]'"
echo ""
info "Install in development mode:"
echo "  pip install -e '.[enterprise]'"
echo ""
