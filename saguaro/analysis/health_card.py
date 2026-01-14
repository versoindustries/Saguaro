import os
import logging
import numpy as np
from typing import Dict, Any

from saguaro.sentinel.engines.external import VultureEngine, MypyEngine
from saguaro.indexing.auto_scaler import get_repo_stats_and_config

logger = logging.getLogger(__name__)


class RepoHealthCard:
    """
    Aggregates 'Texture' metrics for the repository:
    - Complexity Density (Entropy/QWT)
    - Dead Code Ratio
    - Type Safety Score
    - Technical Debt Index
    """

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.stats = get_repo_stats_and_config(self.repo_path)

    def generate_card(self) -> Dict[str, Any]:
        """
        Run analysis and return a consolidated health card.
        """
        card = {
            "loc": self.stats.get("loc", 0),
            "files": self.stats.get("file_count", 0),
            "metrics": {},
        }

        # 1. Complexity Density
        # We use a sampling approach for speed
        card["metrics"]["complexity"] = self._calculate_complexity()

        # 2. Dead Code Ratio
        card["metrics"]["dead_code"] = self._calculate_dead_code()

        # 3. Type Safety
        card["metrics"]["type_safety"] = self._calculate_type_safety()

        # 4. Overall Health Score (0.0 - 1.0)
        card["health_score"] = self._aggregate_score(card["metrics"])

        return card

    def _calculate_complexity(self) -> Dict[str, float]:
        """
        Calculate complexity using Quantum Wavelet Transform (QWT) tokenization density
        or indentation entropy as proxy.
        """
        # Sample files
        sample_files = []
        total_entropy = 0.0
        file_count = 0

        exclusions = [".saguaro", ".git", "venv", "__pycache__", "node_modules"]

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in exclusions]
            for f in files:
                if f.endswith((".py", ".cc", ".js", ".ts")):
                    sample_files.append(os.path.join(root, f))

        # Limit sample
        import random

        if len(sample_files) > 50:
            sample_files = random.sample(sample_files, 50)

        for fpath in sample_files:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content:
                    continue

                # Indentation Entropy as pragmatic complexity proxy
                # (QWT is better but op setup is heavier, let's use fast proxy for Phase 4 prototype)
                lines = content.splitlines()
                if not lines:
                    continue

                indents = []
                for line in lines:
                    stripped = line.lstrip()
                    if stripped:
                        indents.append(len(line) - len(stripped))

                if indents:
                    # Normalized entropy of indentation distribution
                    # Higher variation = higher nesting/complexity
                    arr = np.array(indents)
                    std_dev = np.std(arr)
                    # Normalize roughly 0-10 scale
                    entropy = min(1.0, std_dev / 8.0)
                    total_entropy += entropy
                    file_count += 1
            except Exception:
                pass

        avg_complexity = total_entropy / max(1, file_count)
        return {
            "score": avg_complexity,  # 0.0 (Simple) -> 1.0 (Complex)
            "rating": "High"
            if avg_complexity > 0.7
            else "Medium"
            if avg_complexity > 0.4
            else "Low",
        }

    def _calculate_dead_code(self) -> Dict[str, Any]:
        """
        Run Vulture to estimate dead code ratio.
        """
        engine = VultureEngine(self.repo_path)
        violations = engine.run()  # Returns list of dead code instances

        dead_lines = len(violations)
        total_lines = max(1, self.stats.get("loc", 1))

        ratio = min(
            1.0, (dead_lines * 5) / total_lines
        )  # Assume avg dead chunk is 5 lines?

        return {
            "count": dead_lines,
            "ratio": ratio,
            "score": 1.0 - ratio,  # Higher is better
        }

    def _calculate_type_safety(self) -> Dict[str, Any]:
        """
        Run Mypy to estimate type safety.
        """
        engine = MypyEngine(self.repo_path)
        violations = engine.run()

        error_count = len(violations)
        total_lines = max(1, self.stats.get("loc", 1))

        # 1 error per 100 lines is okay?
        # density = errors / loc
        density = error_count / total_lines

        # Score: 0 errors = 1.0.
        # > 1% errors = 0.0?
        # Let's say 0.01 density is 0.5 score.
        import math

        score = math.exp(-density * 100)  # decay function

        return {"errors": error_count, "density": density, "score": score}

    def _aggregate_score(self, metrics: Dict[str, Any]) -> float:
        """
        Weighted average of component scores.
        """
        # Weights
        w_comp = 0.2
        w_dead = 0.3
        w_type = 0.5

        s_comp = 1.0 - metrics["complexity"]["score"]  # Inverse complexity is health
        s_dead = metrics["dead_code"]["score"]
        s_type = metrics["type_safety"]["score"]

        return (s_comp * w_comp) + (s_dead * w_dead) + (s_type * w_type)
