"""
Knowledge Base
Shared structured storage for agent observations, invariants, and rules.
"""

import os
import json
import time
import logging
from typing import List, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    category: str  # "invariant", "rule", "pattern", "zone"
    key: str
    value: Any
    source: str  # Agent ID or tool
    confidence: float = 1.0
    created_at: float = 0.0


class KnowledgeBase:
    def __init__(self, saguaro_dir: str):
        self.kb_path = os.path.join(saguaro_dir, "knowledge.json")
        self.facts: List[Fact] = []
        self._load()

    def add_fact(
        self,
        category: str,
        key: str,
        value: Any,
        source: str = "user",
        confidence: float = 1.0,
    ):
        # Update existing if key matches?
        # For now, append. Or update if key+category matches.
        existing = next(
            (f for f in self.facts if f.key == key and f.category == category), None
        )
        if existing:
            existing.value = value
            existing.source = source
            existing.confidence = confidence
            existing.created_at = time.time()
        else:
            self.facts.append(
                Fact(
                    category=category,
                    key=key,
                    value=value,
                    source=source,
                    confidence=confidence,
                    created_at=time.time(),
                )
            )
        self._save()

    def get_facts(self, category: str = None) -> List[Fact]:
        if category:
            return [f for f in self.facts if f.category == category]
        return self.facts

    def search(self, query: str) -> List[Fact]:
        results = []
        q = query.lower()
        for f in self.facts:
            if q in f.key.lower() or q in str(f.value).lower():
                results.append(f)
        return results

    def _load(self):
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r") as f:
                    data = json.load(f)
                    self.facts = [Fact(**item) for item in data]
            except Exception as e:
                logger.error(f"Failed to load KB: {e}")
                self.facts = []

    def _save(self):
        try:
            with open(self.kb_path, "w") as f:
                json.dump([asdict(f) for f in self.facts], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save KB: {e}")
