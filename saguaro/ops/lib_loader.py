"""Library loader for SAGUARO native ops.

Delegates to lib_resolver for portable .so discovery.
"""

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def load_saguaro_library():
    """Load the _saguaro_core.so library via TensorFlow."""
    from saguaro.lib_resolver import find_core_library

    try:
        lib_path = find_core_library()
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    try:
        return tf.load_op_library(str(lib_path))
    except Exception as e:
        logger.error(f"Failed to load Saguaro Core library from {lib_path}: {e}")
        raise
