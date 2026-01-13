
import numpy as np
from saguaro.storage.vector_store import VectorStore

def test_hybrid_search():
    store = VectorStore(".saguaro/vectors", dim=10)
    # Mock data
    store.clear()
    
    # Add Item A
    vec_a = np.zeros(10)
    vec_a[0] = 1.0 
    store.add(vec_a, {"name": "CoreClass", "file": "core.py", "type": "class"})
    
    # Add Item B
    vec_b = np.copy(vec_a) # Identical vector
    store.add(vec_b, {"name": "UtilityClass", "file": "util.py", "type": "class"})
    
    query = np.zeros(10)
    query[0] = 1.0 # Perfect match for both
    
    # 1. Normal Query
    res1 = store.query(query, k=2)
    print("Normal:", [r['name'] for r in res1])
    
    # 2. Boosted Query (Boost UtilityClass)
    boost = {"UtilityClass": 1.0, "CoreClass": 0.0}
    res2 = store.query(query, k=2, boost_map=boost)
    print("Boosted:", [r['name'] for r in res2])
    print("Scores:", [r['score'] for r in res2])

if __name__ == "__main__":
    test_hybrid_search()
