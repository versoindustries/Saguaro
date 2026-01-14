import logging
import os
from typing import List, Dict, Any
from .base import BaseEngine
from ...chronicle.storage import ChronicleStorage
from ...chronicle.diff import SemanticDiff
from saguaro.indexing.engine import IndexEngine
from saguaro.indexing.auto_scaler import get_repo_stats_and_config
from saguaro.tokenization.vocab import CoherenceManager

logger = logging.getLogger(__name__)


class SemanticEngine(BaseEngine):
    """
    Checks for semantic drift and other quantum/holographic violations.
    """

    def __init__(self, repo_path: str):
        super().__init__(repo_path)
        self.storage = ChronicleStorage()
        self.drift_threshold = 0.2

        # Initialize Tokenizer (Coherence)
        self.coherence = CoherenceManager()
        self.coherence.initialize()

    def set_policy(self, config: Dict[str, Any]):
        super().set_policy(config)
        self.drift_threshold = float(config.get("drift_tolerance", 0.2))

    def run(self) -> List[Dict[str, Any]]:
        logger.info("Running SemanticEngine...")
        violations = []

        # 1. Drift Detection
        latest_snapshot = self.storage.get_latest_snapshot()

        if not latest_snapshot:
            # No baseline, so can't calculate drift.
            logger.info("No semantic baseline found. Skipping drift check.")
            return violations

        # Calculate Real State from current codebase
        current_state_blob = self._calculate_current_state()

        drift, details = SemanticDiff.calculate_drift(
            latest_snapshot["hd_state_blob"], current_state_blob
        )

        logger.info(f"Semantic Drift: {drift:.4f}")

        if drift > self.drift_threshold:
            violations.append(
                {
                    "file": "chronicle.db",  # logical file
                    "line": 0,
                    "rule_id": "SEMANTIC-DRIFT",
                    "message": f"Semantic drift {drift:.2f} exceeds threshold {self.drift_threshold}. {SemanticDiff.human_readable_report(drift)}",
                    "severity": "warning",
                    "context": f"Baseline: Snapshot #{latest_snapshot['id']} ({latest_snapshot['description']})",
                }
            )

        return violations

    def _calculate_current_state(self) -> bytes:
        """
        Compute the holographic state of the current working directory.
        Uses IndexEngine logic in-memory (no disk write).
        """
        root_dir = self.repo_path
        saguaro_dir = os.path.join(root_dir, ".saguaro")

        if not os.path.exists(saguaro_dir):
            return b""  # Empty state if not init

        try:
            # Get stats/config
            stats = get_repo_stats_and_config(root_dir)
            engine = IndexEngine(root_dir, saguaro_dir, stats)

            # Use the new optimized compute_state method
            # This handles scanning and processing without persistence
            return engine.compute_state()

        except Exception as e:
            logger.error(f"Failed to calculate current state: {e}")
            return b""
