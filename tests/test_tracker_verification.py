import sys
import os
import time
import pytest
import shutil

# Ensure we can import saguaro
sys.path.append(os.getcwd())

from saguaro.indexing.tracker import IndexTracker

@pytest.fixture
def temp_tracker(tmp_path):
    tracking_file = tmp_path / "tracking.json"
    return IndexTracker(str(tracking_file))

@pytest.fixture
def temp_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("print('hello')")
    return str(f)

def test_tracker_verification_logic(temp_tracker, temp_file):
    # 1. Initial state: not verified
    assert not temp_tracker.is_verified(temp_file)
    
    # 2. Mark as verified
    temp_tracker.update_verification(temp_file, True)
    assert temp_tracker.is_verified(temp_file)
    
    # 3. Modify file: should no longer be verified
    time.sleep(0.1) # Ensure mtime changes
    with open(temp_file, "w") as f:
        f.write("print('goodbye')")
        
    # is_verified should check hash and return False
    assert not temp_tracker.is_verified(temp_file)
    
    # 4. Update index (mtime change, verified=False)
    temp_tracker.update([temp_file])
    assert not temp_tracker.is_verified(temp_file)
    
    # 5. Revivify verification
    temp_tracker.update_verification(temp_file, True)
    assert temp_tracker.is_verified(temp_file)
