"""
Impact Analyzer
Determines the downstream impact of code changes on tests, interfaces, and build targets.
"""

import os
import logging
from typing import List, Dict
from saguaro.refactor.planner import RefactorPlanner

logger = logging.getLogger(__name__)


class ImpactAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.planner = RefactorPlanner(repo_path)

    def analyze_change(self, file_path: str, symbol: str = None) -> Dict:
        """
        Analyze impact of changing a specific file or symbol.
        """
        # 1. Build Dependency Graph
        # We need a graph of the whole repo or at least relevant parts.
        # planner._build_dependency_graph expects a list of files to check.
        # We should scan imports in all files to see who imports 'file_path'.

        # Heuristic: Scan all .py files for imports of the module corresponding to file_path
        target_module = self._file_to_module(file_path)
        dependents = self._find_importers(target_module)

        # 2. Categorize Impacts
        tests = [f for f in dependents if "test" in f or "tests" in f.split(os.sep)]
        interfaces = [
            f for f in dependents if f not in tests
        ]  # Source code dependdents

        # 3. Build Targets
        build_targets = self._find_build_targets(file_path)

        return {
            "target": file_path,
            "module": target_module,
            "impact_score": len(dependents),
            "tests_impacted": tests,
            "interfaces_impacted": interfaces,
            "build_targets": build_targets,
        }

    def _file_to_module(self, path: str) -> str:
        rel = os.path.relpath(path, self.repo_path)
        if rel.endswith(".py"):
            rel = rel[:-3]
        return rel.replace(os.sep, ".")

    def _find_importers(self, module_name: str) -> List[str]:
        importers = []
        # Naive scan
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root or "venv" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath, "r") as f:
                            content = f.read()
                            # Text search for import
                            if (
                                f"import {module_name}" in content
                                or f"from {module_name}" in content
                            ):
                                importers.append(fpath)
                    except Exception:
                        pass
        return importers

    def _find_build_targets(self, file_path: str) -> List[str]:
        """Finds build configuration files in the directory hierarchy."""
        targets = []
        current = os.path.dirname(file_path)
        while current.startswith(self.repo_path):
            # Check for common build files
            for bf in [
                "setup.py",
                "CMakeLists.txt",
                "package.json",
                "Makefile",
                "BUILD",
                "pyproject.toml",
            ]:
                p = os.path.join(current, bf)
                if os.path.exists(p):
                    targets.append(p)
            current = os.path.dirname(current)
        return targets
