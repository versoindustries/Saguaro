"""Library loader for SAGUARO native ops."""
import tensorflow as tf
from pathlib import Path

def load_saguaro_library():
    """Load the _saguaro_core.so library."""
    # Build directory structure assumption:
    # saguaro_proposal/
    #   python/saguaro/ops/lib_loader.py
    #   build/
    #     _saguaro_core.so (or similar location)
    
    # Try finding it relative to this file
    base_path = Path(__file__).resolve().parent.parent.parent
    lib_path = base_path / "build" / "_saguaro_core.so"
    
    if not lib_path.exists():
         # Fallback for installed package scenarios (simplified)
        lib_path = Path(__file__).resolve().parent / "_saguaro_core.so"
        
    if not lib_path.exists():
        raise RuntimeError(f"Could not find _saguaro_core.so at {lib_path}")

    return tf.load_op_library(str(lib_path))
