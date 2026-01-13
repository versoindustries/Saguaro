#!/bin/bash
# saguaro_proposal/build_secure.sh
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Enterprise-grade secure build script for SAGUARO Core.
# Adapted from HighNoon build_secure.sh.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/bin"
LOG_DIR="${BUILD_DIR}/logs"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# SAGUARO Default settings
BUILD_TYPE="Debug"
PRODUCTION_BUILD="OFF"
ENABLE_ANTIDEBUG="OFF"
STRIP_SYMBOLS="OFF"
ENABLE_LTO="OFF"
ENABLE_OLLVM="OFF"
TARGET_PLATFORM="${HSMN_TARGET_PLATFORM:-general}"
ARM_TOOLCHAIN_PREFIX="${HSMN_ARM_TOOLCHAIN_PREFIX:-aarch64-linux-gnu-}"

# Force Enterprise Edition for SAGUARO
HN_EDITION=2
HN_EDITION_NAME="ENTERPRISE"
HN_EDITION_DESCRIPTION="Enterprise tier - source code access + unlimited"

# Detected settings
CXX_COMPILER="${CXX_COMPILER:-g++}"
PYTHON_EXEC=""
CPU_OPT_FLAGS=""
OPENMP_FLAGS=""
PARALLEL_DEFINES=""
TBB_LFLAGS=""
VERSO_TARGET_ARCH=""
TF_XLA_INCLUDE_PATH=""

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "--- $1 ---"
}

log_error() {
    echo >&2 "Error: $1"
}

# =============================================================================
# Detect CPU Vendor
# =============================================================================

detect_cpu_vendor() {
    if [[ -n "${HSMN_CPU_VENDOR:-}" ]]; then
        echo "${HSMN_CPU_VENDOR}"
        return
    fi
    local vendor=""
    if command -v lscpu >/dev/null 2>&1; then
        vendor=$(LC_ALL=C lscpu | awk -F: '/Vendor ID:/ {gsub(/^[ \t]+/, "", $2); print tolower($2); exit}')
    fi
    if [[ -z "${vendor}" && -f /proc/cpuinfo ]]; then
        vendor=$(awk -F: '/vendor_id/ {gsub(/^[ \t]+/, "", $2); print tolower($2); exit}' /proc/cpuinfo)
    fi
    if [[ -z "${vendor}" && "$(uname -s)" == "Darwin" ]]; then
        if command -v sysctl >/dev/null 2>&1; then
            vendor=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr '[:upper:]' '[:lower:]')
        fi
    fi
    vendor=${vendor:-unknown}
    export HSMN_CPU_VENDOR="${vendor}"
    echo "${vendor}"
}

# =============================================================================
# Configure CPU Flags
# =============================================================================

configure_cpu_flags() {
    if [[ -n "${CPU_OPT_FLAGS:-}" ]]; then
        log_info "Using custom CPU flags: ${CPU_OPT_FLAGS}"
        return
    fi

    local arch
    arch=$(uname -m)

    if [[ "${TARGET_PLATFORM}" == "marlin_arm" ]]; then
        CPU_OPT_FLAGS="-march=armv8-a -mtune=cortex-a72"
        export CPU_OPT_FLAGS
        log_info "Cross-compiling for Marlin ARM: ${CPU_OPT_FLAGS}"
        return
    fi

    if [[ "${arch}" == "aarch64" || "${arch}" == "arm64" ]]; then
        CPU_OPT_FLAGS="-march=armv8.2-a+simd+fp16"
        export CPU_OPT_FLAGS
        log_info "Detected ARM64 host. Using CPU flags: ${CPU_OPT_FLAGS}"
        return
    fi

    local flags=""
    if command -v lscpu >/dev/null 2>&1; then
        flags=$(LC_ALL=C lscpu | awk -F: '/Flags:/ {gsub(/^[ \t]+/, "", $2); print tolower($2); exit}')
    fi
    if [[ -z "${flags}" && -f /proc/cpuinfo ]]; then
        flags=$(awk -F: '/flags/ {gsub(/^[ \t]+/, "", $2); print tolower($2); exit}' /proc/cpuinfo)
    fi

    flags=" ${flags} "
    if [[ "${flags}" == *" avx512f "* ]]; then
        CPU_OPT_FLAGS="-mavx512f -mavx512bw -mfma"
        log_info "SIMD: AVX-512 detected"
    elif [[ "${flags}" == *" avx2 "* ]]; then
        CPU_OPT_FLAGS="-mavx2 -mfma"
        log_info "SIMD: AVX2 detected"
    elif [[ "${flags}" == *" avx "* ]]; then
        CPU_OPT_FLAGS="-mavx"
        log_info "SIMD: AVX detected"
    else
        CPU_OPT_FLAGS="-msse4.2"
        log_info "SIMD: SSE4.2 fallback"
    fi
    export CPU_OPT_FLAGS
}

# =============================================================================
# Configure Parallelism (OpenMP / TBB)
# =============================================================================

configure_parallelism() {
    local vendor enable_openmp enable_tbb
    vendor=$(detect_cpu_vendor)
    log_info "Detected CPU vendor: ${vendor}"

    local backend="${HSMN_PAR_BACKEND:-}"
    if [[ -z "${backend}" ]]; then
        if [[ "${vendor}" == *"amd"* || "${vendor}" == *"arm"* || "${vendor}" == *"apple"* ]]; then
            backend="openmp"
        else
            backend="tbb"
        fi
    fi
    export HSMN_PAR_BACKEND="${backend}"

    if [[ -z "${HSMN_ENABLE_OPENMP+x}" ]]; then
        enable_openmp=1
    else
        enable_openmp="${HSMN_ENABLE_OPENMP}"
    fi

    if [[ -z "${HSMN_ENABLE_TBB+x}" ]]; then
        if [[ "${vendor}" == *"amd"* || "${vendor}" == *"arm"* || "${vendor}" == *"apple"* ]]; then
            enable_tbb=0
        else
            enable_tbb=1
        fi
    else
        enable_tbb="${HSMN_ENABLE_TBB}"
    fi

    export HSMN_ENABLE_OPENMP="${enable_openmp}"
    export HSMN_ENABLE_TBB="${enable_tbb}"

    PARALLEL_DEFINES=""
    if [[ "${enable_openmp}" == "1" ]]; then
        OPENMP_FLAGS="${HSMN_OPENMP_FLAGS:--fopenmp}"
        PARALLEL_DEFINES+=" -DHSMN_WITH_OPENMP=1"
        log_info "OpenMP: Enabled (${OPENMP_FLAGS})"
    else
        OPENMP_FLAGS=""
        PARALLEL_DEFINES+=" -DHSMN_WITH_OPENMP=0"
        log_info "OpenMP: Disabled"
    fi

    if [[ "${enable_tbb}" == "1" ]]; then
        PARALLEL_DEFINES+=" -DHSMN_WITH_TBB=1"
        TBB_LFLAGS="-ltbb"
        log_info "Intel TBB: Enabled"
    else
        PARALLEL_DEFINES+=" -DHSMN_WITH_TBB=0"
        TBB_LFLAGS=""
        log_info "Intel TBB: Disabled"
    fi

    log_info "Parallel backend: ${backend}"
}

# =============================================================================
# Detect Python/TensorFlow Environment
# =============================================================================

detect_environment() {
    log_info "Detecting Python and TensorFlow environment"

    if [ -n "${VIRTUAL_ENV-}" ]; then
        PYTHON_EXEC="${VIRTUAL_ENV}/bin/python"
        log_info "Virtual environment detected: ${PYTHON_EXEC}"
    elif [ -f "${PROJECT_ROOT}/venv/bin/python" ]; then
        PYTHON_EXEC="${PROJECT_ROOT}/venv/bin/python"
        log_info "Using project venv: ${PYTHON_EXEC}"
    else
        PYTHON_EXEC="python3"
        log_info "Using system Python: ${PYTHON_EXEC}"
    fi

    if ! command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
        log_error "Python interpreter not found at '${PYTHON_EXEC}'"
        return 1
    fi

    "${PYTHON_EXEC}" -c "import tensorflow" 2>/dev/null || {
        log_error "TensorFlow not found. Please install it."
        return 1
    }

    # Get TensorFlow XLA include path for controller headers
    TF_XLA_INCLUDE_PATH=$("${PYTHON_EXEC}" -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    export TF_XLA_INCLUDE_PATH
    log_info "TensorFlow include: ${TF_XLA_INCLUDE_PATH}"

    # Detect architecture string
    VERSO_TARGET_ARCH=$(uname -m)
    export VERSO_TARGET_ARCH
    log_info "Target architecture: ${VERSO_TARGET_ARCH}"
}

# =============================================================================
# Configure Compiler
# =============================================================================

detect_compiler() {
    if [[ "${TARGET_PLATFORM}" == "marlin_arm" ]]; then
        CXX_COMPILER="${HSMN_ARM_CXX:-${ARM_TOOLCHAIN_PREFIX}g++}"
        log_info "Cross-compiler: ${CXX_COMPILER}"
        return
    fi

    if [[ -z "${CXX_COMPILER:-}" ]]; then
        if command -v clang++ &>/dev/null; then
            CXX_COMPILER="clang++"
        else
            CXX_COMPILER="g++"
        fi
    fi
    log_info "Compiler: ${CXX_COMPILER}"
}

# =============================================================================
# Secrets (Stubbed for SAGUARO)
# =============================================================================
CHAIN_SECRET_HIGH="0x0ULL"
CHAIN_SECRET_LOW="0x0ULL"
HN_CRYPTO_KEY="saguaro_dev_key"

# =============================================================================
# Execution
# =============================================================================

# Run detection
detect_environment
detect_compiler
configure_cpu_flags
configure_parallelism

# Detect final architecture
ARCH=$(uname -m)
case "${ARCH}" in
    x86_64|amd64)
        ARCH_DIR="x86_64"
        echo "📍 Architecture: x86_64"
        ;;
    aarch64|arm64)
        ARCH_DIR="arm64"
        echo "📍 Architecture: arm64"
        ;;
    *)
        echo "❌ Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

echo "🔧 Compiler: ${CXX_COMPILER}"
echo "⚡ CPU Flags: ${CPU_OPT_FLAGS}"
echo "🧵 OpenMP: ${OPENMP_FLAGS:-disabled}"
echo ""

# =============================================================================
# Configure and Build with CMake
# =============================================================================

echo "🔨 Configuring build..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${LOG_DIR}"
cd "${BUILD_DIR}"

# Export environment for CMake
export PYTHON_EXEC
export CPU_OPT_FLAGS
export OPENMP_FLAGS
export TBB_LFLAGS
export PARALLEL_DEFINES
export TF_XLA_INCLUDE_PATH

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
    -DENABLE_LTO="${ENABLE_LTO}" \
    -DSTRIP_SYMBOLS="${STRIP_SYMBOLS}" \
    -DPRODUCTION_BUILD="${PRODUCTION_BUILD}" \
    -DENABLE_ANTIDEBUG="${ENABLE_ANTIDEBUG}" \
    -DCHAIN_SECRET_HIGH="${CHAIN_SECRET_HIGH}" \
    -DCHAIN_SECRET_LOW="${CHAIN_SECRET_LOW}" \
    -DHN_CRYPTO_KEY="${HN_CRYPTO_KEY}" \
    -DHN_EDITION="${HN_EDITION}" \
    -DHN_EDITION_NAME="${HN_EDITION_NAME}" \
    -DCPU_OPT_FLAGS="${CPU_OPT_FLAGS}" \
    -DENABLE_OPENMP="${HSMN_ENABLE_OPENMP:-1}" \
    -DENABLE_TBB="${HSMN_ENABLE_TBB:-0}" \
    -DPython3_EXECUTABLE="${PYTHON_EXEC}" \
    2>&1 | tee cmake_configure.log

echo ""
echo "🔨 Building (this may take a few minutes)..."
cmake --build . --parallel "$(nproc)" 2>&1 | tee cmake_build.log

# =============================================================================
# Verification
# =============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "                          Build Verification                                "
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

BINARY="${BUILD_DIR}/_saguaro_core.so"

if [ -f "${BINARY}" ]; then
    echo "✅ Binary created: ${BINARY}"

    # File size
    SIZE=$(stat -c%s "${BINARY}")
    SIZE_HUMAN=$(numfmt --to=iec "${SIZE}")
    echo "📦 Binary size: ${SIZE_HUMAN}"

    # Check symbols
    echo ""
    echo "🔍 Symbol check:"
    if nm "${BINARY}" 2>&1 | head -1 | grep -q "no symbols"; then
        echo "   ✅ Symbols stripped successfully"
    else
        SYMBOL_COUNT=$(nm "${BINARY}" 2>/dev/null | wc -l || echo "0")
        if [ "${SYMBOL_COUNT}" -gt 0 ]; then
            echo "   ⚠️  Warning: ${SYMBOL_COUNT} symbols remain"
        else
            echo "   ✅ Symbols stripped"
        fi
    fi

    # TensorFlow ops
    echo ""
    echo "🔍 TensorFlow op check:"
    OP_COUNT=$(nm -D "${BINARY}" 2>/dev/null | grep -c "RegisterOp\|OpKernel" || echo "0")
    if [ "${OP_COUNT}" -gt 0 ]; then
        echo "   ✅ TensorFlow operations detected"
    else
        echo "   ⚠️  Warning: No TensorFlow ops detected in dynamic symbols"
    fi

else
    echo "❌ Build failed - binary not found at ${BINARY}"
    echo ""
    echo "Check logs:"
    echo "   ${BUILD_DIR}/cmake_configure.log"
    echo "   ${BUILD_DIR}/cmake_build.log"
    exit 1
fi

echo "Done!"
