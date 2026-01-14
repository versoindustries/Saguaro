import sys
import os
import logging

# Ensure we can import saguaro
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saguaro.ops.quantum_ops import load_saguaro_core

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_load_ops():
    logger.info("Testing SAGUARO Core Op Loading...")
    try:
        module = load_saguaro_core()
        logger.info(f"Successfully loaded library: {module}")

        # List available ops in the module
        logger.info("Available Ops:")
        # dir(module) shows the generated functions
        for item in dir(module):
            if not item.startswith("_"):
                logger.info(f" - {item}")

    except Exception as e:
        logger.error(f"FAILED to load ops: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_load_ops()
