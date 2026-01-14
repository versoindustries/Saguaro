"""
Verification Contract Manager
Handles API contract verification, semantic diff classification, and invariant checking.
"""

import ast
from typing import List


class ContractVerifier:
    def verify_api_compat(self, old_code: str, new_code: str) -> List[str]:
        """Checks for breaking API changes."""
        violations = []
        old_tree = ast.parse(old_code)
        new_tree = ast.parse(new_code)

        old_funcs = {
            node.name: node
            for node in ast.walk(old_tree)
            if isinstance(node, ast.FunctionDef)
        }
        new_funcs = {
            node.name: node
            for node in ast.walk(new_tree)
            if isinstance(node, ast.FunctionDef)
        }

        for name, old_node in old_funcs.items():
            if name not in new_funcs:
                violations.append(f"Function {name} removed")
                continue

            new_node = new_funcs[name]
            # Check args
            if len(old_node.args.args) < len(new_node.args.args):
                # Only breaking if new args are not optional (have defaults)
                # This is a simplification
                pass

        return violations


class SemanticDiffClassifier:
    def classify(self, diff: str) -> str:
        """Classifies a diff as refactor, behavior_change, or fix."""
        # Stub for prototype
        if "test" in diff:
            return "test_update"
        if "def " in diff:
            return "logic_change"
        return "refactor"


class InvariantChecker:
    def check_golden_rules(self, code: str) -> List[str]:
        """Checks code against golden invariants (no new deps, etc)."""
        violations = []
        if "import telnetlib" in code:
            violations.append("Banned import: telnetlib")
        # etc
        return violations
