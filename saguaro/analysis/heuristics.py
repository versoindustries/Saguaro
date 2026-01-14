import os
import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HeuristicAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        # Heuristic Defaults
        self.reflection_patterns = [
            r"getattr\(",
            r"setattr\(",
            r"__import__\(",
            r"importlib\.",
            r"pydoc\.locate",
        ]
        self.config_patterns = [
            r"config",
            r"settings",
            r"setup",
            r"env",
            r"json",
            r"yaml",
            r"toml",
        ]

    def check_heuristics(self, file_path: str) -> Dict[str, bool]:
        """
        Checks if a file matches 'unsafe to delete' heuristics.
        """
        report = {"safe": True, "reasons": []}

        try:
            with open(file_path, "r", errors="ignore") as f:
                content = f.read()

            # 1. Reflection Check
            if any(re.search(p, content) for p in self.reflection_patterns):
                report["safe"] = False
                report["reasons"].append("Uses reflection/dynamic imports")

            # 2. Entry Point / Config Check
            if any(
                p in os.path.basename(file_path).lower() for p in self.config_patterns
            ):
                report["safe"] = False
                report["reasons"].append("Config-like filename")

            # 3. Main/Entry check
            if 'if __name__ == "__main__":' in content or "def main():" in content:
                report["safe"] = False
                report["reasons"].append("Possible Entry Point (main)")

            return report
        except Exception as e:
            logger.warning(f"Heuristic check failed for {file_path}: {e}")
            return {"safe": False, "reasons": ["Analysis Error"]}
