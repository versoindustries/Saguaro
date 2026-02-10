"""
SAGUARO Unified Native Library Resolver

Single source of truth for finding _saguaro_core.so and _saguaro_native.so.
Works in both development (source tree) and installed (site-packages) contexts.

Search Order:
    1. SAGUARO_LIB_DIR environment variable (explicit override)
    2. Package directory (saguaro/ — for pip-installed wheels)
    3. Adjacent build/ directory (for CMake development builds)
    4. LD_LIBRARY_PATH / DYLD_LIBRARY_PATH scan (system fallback)
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Package root: the `saguaro/` directory containing this file
_PACKAGE_DIR = Path(__file__).resolve().parent

# Development project root: parent of saguaro/ (only meaningful in source tree)
_PROJECT_ROOT = _PACKAGE_DIR.parent


def _get_env_lib_dir() -> Optional[Path]:
    """Check for explicit library directory override."""
    env_dir = os.environ.get("SAGUARO_LIB_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
        logger.warning(f"SAGUARO_LIB_DIR={env_dir} is not a valid directory")
    return None


def _get_ld_library_paths():
    """Yield directories from LD_LIBRARY_PATH / DYLD_LIBRARY_PATH."""
    for env_var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        val = os.environ.get(env_var, "")
        for entry in val.split(os.pathsep):
            entry = entry.strip()
            if entry and os.path.isdir(entry):
                yield Path(entry)


def find_native_library(name: str) -> Path:
    """Find a native shared library by filename.

    Args:
        name: Library filename (e.g., '_saguaro_core.so' or '_saguaro_native.so')

    Returns:
        Resolved absolute Path to the library file.

    Raises:
        FileNotFoundError: If the library cannot be found in any search location.
    """
    searched = []

    # --- Priority 1: Environment variable override ---
    env_dir = _get_env_lib_dir()
    if env_dir:
        candidate = env_dir / name
        searched.append(str(candidate))
        if candidate.exists():
            logger.debug(f"Found {name} via SAGUARO_LIB_DIR: {candidate}")
            return candidate

    # --- Priority 2: Package directory (pip-installed / in-tree) ---
    # This is the canonical location when Saguaro is pip-installed as a wheel.
    # The .so files live alongside the Python modules in saguaro/.
    pkg_candidates = [
        _PACKAGE_DIR / name,                          # saguaro/<name>
        _PACKAGE_DIR / "ops" / name,                  # saguaro/ops/<name>
        _PACKAGE_DIR / "native" / "bin" / "x86_64" / name,  # dev native build
        _PACKAGE_DIR / "native" / "build" / name,     # dev native cmake
    ]
    for candidate in pkg_candidates:
        searched.append(str(candidate))
        if candidate.exists():
            logger.debug(f"Found {name} in package directory: {candidate}")
            return candidate

    # --- Priority 3: Adjacent build/ directory (development CMake builds) ---
    dev_candidates = [
        _PROJECT_ROOT / "build" / name,               # <repo>/build/<name>
        _PROJECT_ROOT / name,                         # <repo>/<name>
    ]
    for candidate in dev_candidates:
        searched.append(str(candidate))
        if candidate.exists():
            logger.debug(f"Found {name} in development build: {candidate}")
            return candidate

    # --- Priority 4: System library paths ---
    for lib_dir in _get_ld_library_paths():
        candidate = lib_dir / name
        searched.append(str(candidate))
        if candidate.exists():
            logger.debug(f"Found {name} in system library path: {candidate}")
            return candidate

    raise FileNotFoundError(
        f"Could not find native library '{name}'.\n"
        f"Searched locations:\n"
        + "\n".join(f"  - {s}" for s in searched)
        + "\n\nTo fix this, either:\n"
        "  1. Build with CMake and ensure .so files are in saguaro/ or build/\n"
        "  2. Set SAGUARO_LIB_DIR to the directory containing the .so files\n"
        "  3. Install Saguaro with: pip install -e . (from the repo root)"
    )


def find_core_library() -> Path:
    """Find the TensorFlow-linked _saguaro_core.so library."""
    return find_native_library("_saguaro_core.so")


def find_native_only_library() -> Path:
    """Find the TF-free _saguaro_native.so library.

    Falls back to _saguaro_core.so if the native-only build is not available.
    """
    try:
        return find_native_library("_saguaro_native.so")
    except FileNotFoundError:
        logger.info(
            "_saguaro_native.so not found, falling back to _saguaro_core.so"
        )
        return find_native_library("_saguaro_core.so")
