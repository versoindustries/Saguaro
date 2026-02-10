import os
import json
import logging
import hashlib
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class IndexTracker:
    """
    Tracks file modification times and verification states to support 
    incremental indexing and robust governance.
    """

    def __init__(self, tracking_file: str):
        self.tracking_file = tracking_file
        self.state: Dict[str, Dict[str, Any]] = {}  # path -> {mtime, hash, verified}
        self._load()

    def _load(self):
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, "r") as f:
                    raw_state = json.load(f)
                    
                    # Handle legacy format (flat dict of mtimes)
                    if raw_state and not isinstance(next(iter(raw_state.values())), dict):
                        logger.info("Upgrading IndexTracker state to new format.")
                        self.state = {
                            path: {"mtime": mtime, "hash": "", "verified": False}
                            for path, mtime in raw_state.items()
                        }
                    else:
                        self.state = raw_state
            except Exception as e:
                logger.warning(f"Failed to load tracking file: {e}. Starting fresh.")
                self.state = {}
        else:
            self.state = {}

    def save(self):
        try:
            with open(self.tracking_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tracking file: {e}")

    def filter_needs_indexing(self, file_paths: List[str]) -> List[str]:
        """
        Returns list of files that have changed or are new.
        """
        needs_update = []
        for path in file_paths:
            if not os.path.exists(path):
                if path in self.state:
                    del self.state[path]
                continue

            current_mtime = os.path.getmtime(path)
            entry = self.state.get(path, {})
            last_mtime = entry.get("mtime", 0.0)

            if current_mtime > last_mtime:
                needs_update.append(path)

        return needs_update

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        if not os.path.exists(file_path):
            return ""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
            return ""

    def update(self, file_paths: List[str]):
        """
        Updates the indexing state for the given files.
        """
        for path in file_paths:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                file_hash = self._compute_hash(path)
                
                # If mtime or hash changed, it's no longer verified
                existing = self.state.get(path, {})
                was_verified = existing.get("verified", False)
                if existing.get("hash") != file_hash:
                    was_verified = False

                self.state[path] = {
                    "mtime": mtime,
                    "hash": file_hash,
                    "verified": was_verified
                }
        self.save()

    def update_verification(self, file_path: str, verified: bool = True):
        """Marks a file as verified and stores its current hash."""
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            file_hash = self._compute_hash(file_path)
            self.state[file_path] = {
                "mtime": mtime,
                "hash": file_hash,
                "verified": verified
            }
            self.save()

    def is_verified(self, file_path: str) -> bool:
        """Check if a file is verified based on its current content hash."""
        if not os.path.exists(file_path):
            return False
            
        entry = self.state.get(file_path)
        if not entry or not entry.get("verified"):
            return False
            
        # Verify hash match to ensure content hasn't changed since verification
        current_hash = self._compute_hash(file_path)
        return entry.get("hash") == current_hash

    def clear(self):
        self.state = {}
        self.save()
