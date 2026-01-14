"""
SAGUARO Sentinel Verifier
Enforces rules against files.
"""

import os
import logging
from typing import List, Dict, Any
from .engines import BaseEngine, NativeEngine, RuffEngine, MypyEngine, VultureEngine
from .policy import PolicyManager
from .engines.semantic import SemanticEngine

logger = logging.getLogger(__name__)


class SentinelVerifier:
    def __init__(self, repo_path: str, engines: List[str] = None):
        self.repo_path = os.path.abspath(repo_path)
        self.engines: List[BaseEngine] = []
        self.policy = PolicyManager(self.repo_path)

        # Map names to classes
        engine_map = {
            "native": NativeEngine,
            "ruff": RuffEngine,
            "mypy": MypyEngine,
            "vulture": VultureEngine,
            "semantic": SemanticEngine,
        }

        if engines is None:
            # Default to all if not specified
            engines = ["native", "ruff", "mypy", "vulture", "semantic"]

        for name in engines:
            if name in engine_map:
                try:
                    self.engines.append(engine_map[name](self.repo_path))
                except Exception as e:
                    logger.warning(f"Failed to initialize engine {name}: {e}")
            else:
                logger.warning(f"Unknown engine: {name}")

    def verify_all(self, path_arg: str = ".") -> List[Dict[str, Any]]:
        """
        Runs all configured engines and enforces policy.
        """
        all_violations = []

        for engine in self.engines:
            # Inject policy
            engine.set_policy(self.policy.config)

            try:
                logger.info(f"Running engine: {engine.__class__.__name__}")
                violations = engine.run()
                all_violations.extend(violations)
            except Exception as e:
                logger.error(f"Engine {engine.__class__.__name__} failed: {e}")

        # Apply policy
        return self.policy.evaluate(all_violations)
