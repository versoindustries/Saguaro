"""
SAGUARO Auto-Legislator: Governance Discovery Engine
"""

import os
import re
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger("saguaro.legislator")


class Legislator:
    """
    Forensic engine that scans a codebase for implicit patterns
    and drafts explicit governance rules.
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)
        self.draft_file = os.path.join(self.root_dir, ".saguaro.rules.draft")

    def draft_rules(self) -> str:
        """
        Scan codebase and generate a draft rules file.
        Returns the content of the generated yaml.
        """
        logger.info(f"Starting legislative session for {self.root_dir}")

        findings = []

        # 1. Inspector: Docstrings
        # Check if python functions generally have docstrings
        docstring_compliance = self._inspect_docstrings()
        if docstring_compliance["score"] > 0.8:  # Found validation > 80%
            findings.append(
                {
                    "scope": "**/*.py",
                    "rule": "Must have docstrings for all public functions",
                    "evidence": f"Found in {docstring_compliance['compliant']}/{docstring_compliance['total']} functions ({docstring_compliance['score']:.1%})",
                }
            )

        # 2. Inspector: Snake Case
        # simplified check for filenames
        casing_compliance = self._inspect_casing()
        if casing_compliance["score"] > 0.9:
            findings.append(
                {
                    "scope": "**/*",
                    "rule": "Filenames must be snake_case",
                    "evidence": f"Found in {casing_compliance['compliant']}/{casing_compliance['total']} files",
                }
            )

        # Save draft
        draft_content = {
            "version": "1.0-draft",
            "generated_by": "SAGUARO Auto-Legislator",
            "rules": findings,
        }

        with open(self.draft_file, "w") as f:
            yaml.dump(draft_content, f, sort_keys=False)

        logger.info(f"Drafted {len(findings)} rules to {self.draft_file}")
        return yaml.dump(draft_content, sort_keys=False)

    def _inspect_docstrings(self) -> Dict[str, Any]:
        """Simple regex heuristic for docstrings in python files"""
        total_funcs = 0
        docstring_funcs = 0

        for root, _, files in os.walk(self.root_dir):
            if ".git" in root or "node_modules" in root or ".saguaro" in root:
                continue

            for file in files:
                if file.endswith(".py"):
                    try:
                        path = os.path.join(root, file)
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            # Find def foo(...):
                            # funcs = re.findall(r'def\s+([a-zA-Z_]\w*)\s*\(', content)

                            # Naive check: Does it have """ or ''' after?
                            # This is very approximate, a real AST walker would be used in prod
                            # but this proves the 'legislator' concept.
                            lines = content.split("\n")
                            for i, line in enumerate(lines):
                                if line.strip().startswith("def "):
                                    total_funcs += 1
                                    # Look ahead for docstring
                                    for j in range(1, 4):  # Check next 3 lines
                                        if i + j < len(lines) and (
                                            '"""' in lines[i + j]
                                            or "'''" in lines[i + j]
                                        ):
                                            docstring_funcs += 1
                                            break
                    except Exception as e:
                        logger.debug(f"Skipping file {file}: {e}")

        score = docstring_funcs / total_funcs if total_funcs > 0 else 0
        return {"total": total_funcs, "compliant": docstring_funcs, "score": score}

    def _inspect_casing(self) -> Dict[str, Any]:
        """Check if filenames are snake_case"""
        total = 0
        compliant = 0

        camel_pattern = re.compile(r"[A-Z]")

        for root, _, files in os.walk(self.root_dir):
            if ".git" in root or "node_modules" in root or ".saguaro" in root:
                continue
            for file in files:
                total += 1
                if not camel_pattern.search(file) and " " not in file:
                    compliant += 1

        score = compliant / total if total > 0 else 0
        return {"total": total, "compliant": compliant, "score": score}
