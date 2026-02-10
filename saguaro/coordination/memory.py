from typing import Any, Dict, Optional
import json
import os


class SharedMemory:
    """
    Shared Structured Memory for agents.
    Saguaro manages distinct memory namespaces to separate facts from style.
    """

    VALID_TIERS = ["working", "episodic", "semantic", "preference"]

    def __init__(self, persistence_path: str = ".saguaro/shared_memory.json"):
        self.persistence_path = persistence_path
        # self.data structure: { tier: { key: { value: ..., source: ..., timestamp: ... } } }
        self.data: Dict[str, Dict[str, Any]] = {tier: {} for tier in self.VALID_TIERS}
        self._load()

    def write_fact(self, key: str, value: Any, agent_id: str, tier: str = "working"):
        """Writes a fact to a specific memory tier."""
        if tier not in self.VALID_TIERS:
            # Fallback or error? Protocol says these four.
            # We'll default to working if unknown for robustness but log it.
            tier = "working"

        import time

        self.data[tier][key] = {
            "value": value,
            "source": agent_id,
            "timestamp": time.time(),
        }
        self._save()

    def read_fact(self, key: str, tier: str = "working") -> Optional[Any]:
        """Reads a fact from a specific tier."""
        if tier not in self.VALID_TIERS:
            return None
        entry = self.data[tier].get(key)
        return entry["value"] if entry else None

    def list_facts(self, tier: Optional[str] = None) -> Dict[str, Any]:
        """Lists facts, optionally filtered by tier."""
        if tier:
            return self.data.get(tier, {})
        # Return flattened view for backward compatibility or the whole dict?
        # CLI expects a flat dict currently. Let's return the whole dict if no tier.
        return self.data

    def _save(self):
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _load(self):
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r") as f:
                loaded = json.load(f)
                # Migration/Validation: if it's the old flat format, put it in 'working'
                if loaded and "working" not in loaded:
                    self.data["working"] = loaded
                else:
                    self.data.update(loaded)
        except Exception:
            self.data = {tier: {} for tier in self.VALID_TIERS}
