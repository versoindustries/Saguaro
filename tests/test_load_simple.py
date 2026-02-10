import sys
import os
import tensorflow as tf

# Ensure we can import saguaro
sys.path.append(os.getcwd())

from saguaro.ops.quantum_ops import load_saguaro_core

def test_load():
    print("Testing library loading...")
    try:
        mod = load_saguaro_core()
        if mod:
            print("Successfully loaded SAGUARO Core!")
            print(f"Ops: {dir(mod)}")
        else:
            print("Failed to load SAGUARO Core (None returned)")
    except Exception as e:
        print(f"Error loading library: {e}")

if __name__ == "__main__":
    test_load()
