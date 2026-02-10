import numpy as np
import os
import shutil
import sys

# Ensure we can import saguaro
sys.path.append(os.getcwd())

from saguaro.storage.memmap_vector_store import MemoryMappedVectorStore

def test_refutation_boost():
    print("Testing Refutation Boost (Literal Boosting)...")
    storage_path = "/tmp/saguaro_test_boost"
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
    
    # dim=128
    store = MemoryMappedVectorStore(storage_path, dim=128)
    
    # 1. Add two similar vectors
    vec = np.random.rand(128).astype(np.float32)
    # Slightly vary them so one is naturally better
    vec_a = vec + np.random.rand(128).astype(np.float32) * 0.01
    vec_b = vec + np.random.rand(128).astype(np.float32) * 0.01
    
    meta_a = {"file": "core.py", "name": "StandardClass", "type": "class"}
    meta_b = {"file": "legacy.py", "name": "OldClass_v2", "type": "class"}
    
    store.add(vec_a, meta_a)
    store.add(vec_b, meta_b)
    
    # 2. Query WITHOUT query_text
    # They should have similar scores
    query_vec = vec.copy()
    results_no_text = store.query(query_vec, k=2)
    print(f"No text scores: {results_no_text[0]['name']}={results_no_text[0]['score']:.4f}, {results_no_text[1]['name']}={results_no_text[1]['score']:.4f}")
    
    # 3. Query WITH query_text targeting "OldClass_v2"
    results_with_text = store.query(query_vec, k=2, query_text="Refactor OldClass_v2 implementation")
    print(f"With text scores: {results_with_text[0]['name']}={results_with_text[0]['score']:.4f}, {results_with_text[1]['name']}={results_with_text[1]['score']:.4f}")
    
    # Verify that OldClass_v2 is now boosted (likely top rank)
    assert results_with_text[0]["name"] == "OldClass_v2", "Refutation boost failed to promote exact match"
    
    # Cleanup
    store.close()
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
    print("Refutation Boost Test Passed!")

if __name__ == "__main__":
    try:
        test_refutation_boost()
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
