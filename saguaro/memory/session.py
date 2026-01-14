"""
SAGUARO Agent Memory Protocol
Manages session state and history persistence.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AgentSession:
    def __init__(self, session_id: str, repo_path: str):
        self.session_id = session_id
        self.repo_path = repo_path
        self.saguaro_dir = os.path.join(repo_path, ".saguaro")
        self.session_dir = os.path.join(self.saguaro_dir, "sessions")
        self.history: List[Dict[str, Any]] = []

        self._ensure_dir()
        self.load()

    def _ensure_dir(self):
        os.makedirs(self.session_dir, exist_ok=True)

    def _get_file_path(self):
        # Sanitize session_id
        safe_id = "".join(
            [c for c in self.session_id if c.isalnum() or c in ("-", "_")]
        )
        return os.path.join(self.session_dir, f"{safe_id}.json")

    def load(self):
        path = self._get_file_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
            except Exception as e:
                logger.error(f"Failed to load session {self.session_id}: {e}")

    def save(self):
        path = self._get_file_path()
        data = {
            "session_id": self.session_id,
            "last_updated": time.time(),
            "history": self.history,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {self.session_id}: {e}")

    def add_interaction(self, query: str, results: List[Any], tool: str = "query"):
        """Records an agent interaction."""
        entry = {
            "timestamp": time.time(),
            "tool": tool,
            "input": query,
            "output_summary": f"{len(results)} results"
            if isinstance(results, list)
            else "result",
            # We don't store full result payload to save space, or maybe strictly top 1?
            # For now, store minimal
        }
        self.history.append(entry)
        self.save()

    def get_context_window(self, k=5):
        """Returns last k interactions for context."""
        return self.history[-k:]
