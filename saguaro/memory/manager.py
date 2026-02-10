"""
Tiered Agentic Memory for SAGUARO.
Manages Working, Episodic, Semantic, and Preference memory namespaces.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MemoryTier:
    WORKING = "working"     # Current context / task
    EPISODIC = "episodic"   # Past interactions
    SEMANTIC = "semantic"   # Factual knowledge (holographic)
    PREFERENCE = "preference" # User-specific style/rules

class MemoryManager:
    def __init__(self, storage_root: str):
        self.root = os.path.join(storage_root, "agentic_memory")
        os.makedirs(self.root, exist_ok=True)
        
    def _get_path(self, tier: str, namespace: str = "default") -> str:
        tier_dir = os.path.join(self.root, tier)
        os.makedirs(tier_dir, exist_ok=True)
        return os.path.join(tier_dir, f"{namespace}.json")

    def store(self, tier: str, content: Any, namespace: str = "default"):
        path = self._get_path(tier, namespace)
        data = {
            "timestamp": time.time(),
            "content": content
        }
        # In a real implementation, we would append or use a DB
        # For this prototype, we store as list items in JSON
        existing = self.retrieve(tier, namespace) or []
        existing.append(data)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def retrieve(self, tier: str, namespace: str = "default") -> List[Dict[str, Any]]:
        path = self._get_path(tier, namespace)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def clear(self, tier: str, namespace: str = "default"):
        path = self._get_path(tier, namespace)
        if os.path.exists(path):
            os.remove(path)
