import os
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class IndexTracker:
    """
    Tracks file modification times to support incremental indexing.
    """

    def __init__(self, tracking_file: str):
        self.tracking_file = tracking_file
        self.state: Dict[str, float] = {}  # path -> mtime
        self._load()

    def _load(self):
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, "r") as f:
                    self.state = json.load(f)
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
                # File deleted? We should probably handle deletion too,
                # but for now we focus on re-indexing changed files.
                if path in self.state:
                    del self.state[path]  # Remove from tracking
                continue

            current_mtime = os.path.getmtime(path)
            last_mtime = self.state.get(path, 0.0)

            if current_mtime > last_mtime:
                needs_update.append(path)

        return needs_update

    def update(self, file_paths: List[str]):
        """
        Updates the tracking state for the given files (marking them as indexed).
        """
        for path in file_paths:
            if os.path.exists(path):
                self.state[path] = os.path.getmtime(path)
        self.save()

    def clear(self):
        self.state = {}
        self.save()
