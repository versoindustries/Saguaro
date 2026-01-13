from typing import Any, Dict, Optional
import json
import os

class SharedMemory:
    """
    Shared Structured Memory for agents.
    Agents write facts (e.g., "invariants discovered") here, not chat logs.
    """
    
    def __init__(self, persistence_path: str = ".saguaro/shared_memory.json"):
        self.persistence_path = persistence_path
        self.facts: Dict[str, Any] = {}
        self._load()

    def write_fact(self, key: str, value: Any, agent_id: str):
        """Writes a fact to the shared memory."""
        self.facts[key] = {
            "value": value,
            "source": agent_id,
            "timestamp": "TODO: timestamp"
        }
        self._save()

    def read_fact(self, key: str) -> Optional[Any]:
        """Reads a fact."""
        entry = self.facts.get(key)
        return entry["value"] if entry else None

    def list_facts(self) -> Dict[str, Any]:
        return self.facts

    def _save(self):
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, 'w') as f:
            json.dump(self.facts, f, indent=2)

    def _load(self):
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, 'r') as f:
                self.facts = json.load(f)
        except Exception:
            self.facts = {}
