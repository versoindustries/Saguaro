"""Library loader for SAGUARO native ops."""

import tensorflow as tf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_saguaro_library():
    """Load the _saguaro_core.so library."""
    current_dir = Path(__file__).resolve().parent
    # saguaro/ directory (parent of ops/)
    saguaro_dir = current_dir.parent
    # Project root (parent of saguaro/)
    project_root = saguaro_dir.parent

    # Priority search paths - check all likely locations
    candidates = [
        # Priority 1: Project root build directory (CMake output)
        project_root / "build" / "_saguaro_core.so",
        # Priority 2: saguaro package directory
        saguaro_dir / "_saguaro_core.so",
        # Priority 3: Colocated in saguaro/ops
        current_dir / "_saguaro_core.so",
        # Priority 4: Development native build
        saguaro_dir / "native" / "bin" / "x86_64" / "_saguaro_core.so",
        # Priority 5: Native CMake build
        saguaro_dir / "native" / "build" / "_saguaro_core.so",
    ]

    lib_path = None
    for candidate in candidates:
        if candidate.exists():
            lib_path = candidate
            logger.debug(f"Found Saguaro Core at: {lib_path}")
            break

    if lib_path is None:
        logger.error(
            "Could not find _saguaro_core.so in any of the expected locations."
        )
        return None

    try:
        return tf.load_op_library(str(lib_path))
    except Exception as e:
        logger.error(f"Failed to load Saguaro Core library from {lib_path}: {e}")
        raise
