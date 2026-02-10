import numpy as np
import os
import shutil
import sys

# Ensure we can import saguaro
sys.path.append(os.getcwd())

from saguaro.storage.memmap_vector_store import MemoryMappedVectorStore

def test_upsert():
    print("Testing Vector Store Upsert...")
    storage_path = "/tmp/saguaro_test_upsert"
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
    
    store = MemoryMappedVectorStore(storage_path, dim=128)
    
    meta = {"file": "test.py", "name": "foo", "type": "function"}
    vec1 = np.random.rand(128).astype(np.float32)
    
    # 1. Add first time
    store.add(vec1, meta)
    assert len(store) == 1
    
    # 2. Add second time (same file/name)
    vec2 = np.random.rand(128).astype(np.float32)
    store.add(vec2, meta)
    
    # Verify count is still 1 (upsert!)
    assert len(store) == 1, f"Expected count 1, got {len(store)}"
    
    # Verify vector was updated
    results = store.query(vec2, k=1)
    assert results[0]["score"] > 0.99, "Vector was not updated correctly"
    
    # 3. Batch add with duplicates
    meta3 = {"file": "test.py", "name": "bar", "type": "function"}
    vec3 = np.random.rand(128).astype(np.float32)
    
    store.add_batch(
        np.array([vec1, vec3]), 
        [meta, meta3]
    )
    
    # Combined: foo (upserted again), bar (new)
    assert len(store) == 2, f"Expected count 2, got {len(store)}"
    
    # Cleanup
    store.close()
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
    print("Upsert Test Passed!")

if __name__ == "__main__":
    try:
        test_upsert()
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
