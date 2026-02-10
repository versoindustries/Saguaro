import tensorflow as tf
import numpy as np
import os
import sys

# Ensure we can import saguaro
sys.path.append(os.getcwd())

from saguaro.ops.holographic import serialize_bundle, deserialize_bundle

def test_holographic_serialization_roundtrip():
    print("Testing Holographic Serialization Round-trip...")
    
    # 1. Create a random bundle tensor
    dim = 8192
    original_bundle = tf.random.uniform([dim], dtype=tf.float32)
    
    # 2. Serialize
    blob = serialize_bundle(original_bundle)
    assert isinstance(blob, bytes), "Serialization should return bytes"
    
    # 3. Deserialize
    restored_bundle = deserialize_bundle(blob)
    
    # 4. Verify
    # Use a small epsilon for float comparison if needed, but tf.io.serialize should be exact
    diff = tf.reduce_max(tf.abs(original_bundle - restored_bundle))
    print(f"Max difference: {diff.numpy()}")
    
    assert diff.numpy() == 0, "Restored bundle does not match original"
    print("Serialization Round-trip Passed!")

if __name__ == "__main__":
    try:
        test_holographic_serialization_roundtrip()
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
