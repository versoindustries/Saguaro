
import os
import sys
import logging
import tensorflow as tf
import shutil
import fnmatch
import numpy as np

# Ensure we are in the repo root
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from saguaro.indexing.engine import IndexEngine, process_batch_worker
from saguaro.tokenization.vocab import CoherenceManager
from saguaro.indexing.tracker import IndexTracker
from saguaro.utils.file_utils import get_code_files

def test_full_scan_repro():
    print("\n=== Testing Full Scan Indexing (Parallel Arch) ===")
    repo_path = os.getcwd() # Index self
    saguaro_dir = os.path.join(repo_path, ".saguaro_repro")
    
    # Clean previous repro state to ensure fresh start
    if os.path.exists(saguaro_dir):
        shutil.rmtree(saguaro_dir)
    os.makedirs(saguaro_dir, exist_ok=True)
    
    config = {
        'total_dim': 128,
        'active_dim': 64,
        'dark_space_ratio': 0.5,
        'loc': 150000,
        'indexing': {'exclude': []}
    }
    
    try:
        engine = IndexEngine(repo_path, saguaro_dir, config)
        print("IndexEngine initialized.")
        
        # 1. Test Robust File Discovery
        all_files = get_code_files(repo_path, exclusions=[])
        
        total_files = len(all_files)
        print(f"Discovered {total_files} files using new file_utils.")
        
        if total_files == 0:
            print("FAILURE: No files discovered!")
            return

        # 2. Check Tracker Logic
        tracker = IndexTracker(os.path.join(saguaro_dir, "tracking.json"))
        needed = tracker.filter_needs_indexing(all_files)
        print(f"Tracker says {len(needed)} files need indexing.")
        
        # 3. Test Worker Logic (Simulate what CLI does)
        batch = needed[:64] if needed else []
        print(f"Attempting to index first batch of {len(batch)} files via worker...")
        
        if batch:
            # Call worker directly (synch)
            meta, vectors = process_batch_worker(batch, config['active_dim'], config['total_dim'])
            
            if vectors is None:
                 print("Worker returned None.")
            else:
                 print(f"Worker returned {len(meta)} metadata items and vector shape {vectors.shape}")
            
            # Ingest
            f_count, e_count = engine.ingest_worker_result(meta, vectors)
            print(f"Ingested Result: {f_count} files, {e_count} entities")
            
            if f_count == 0:
                 print("FAILURE: Engine returned 0 processed files.")
            else:
                 print("SUCCESS: Engine processed files.")
                 
            engine.commit()
            print("Commit successful.")
            
        else:
             print("FAILURE: No files to index in batch.")

    except Exception as e:
        print(f"Indexing FAILED: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    test_full_scan_repro()
