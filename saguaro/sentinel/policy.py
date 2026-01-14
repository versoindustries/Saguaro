import os
import yaml
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PolicyManager:
    """
    Unified Policy Enforcer for Sentinel.
    Evaluates violations against configured policies.
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.config = self._load_policy()

    def _load_policy(self) -> Dict[str, Any]:
        # Check standard location first
        policy_path = os.path.join(self.repo_path, ".saguaro/policy.yaml")
        if not os.path.exists(policy_path):
            # Check legacy location
            policy_path = os.path.join(self.repo_path, ".saguaro.policy.yaml")

        # Default policy
        default_policy = {
            "strict_mode": False,  # If True, violations = failure
            "drift_tolerance": 0.2,  # Semantic drift limit
            "excluded_rules": [],  # List of rule IDs to ignore
            "auto_fix": True,  # Allow auto-fixes by default
        }

        if os.path.exists(policy_path):
            try:
                with open(policy_path, "r") as f:
                    user_policy = yaml.safe_load(f) or {}
                    # Merge (simple override)
                    default_policy.update(user_policy)
            except Exception as e:
                logger.warning(f"Failed to load policy file {policy_path}: {e}")

        return default_policy

    def evaluate(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process violations: filter based on policy.
        """
        active_violations = []
        rules = self.config.get("excluded_rules")
        excluded = set(rules if rules is not None else [])

        for v in violations:
            rule_id = v.get("rule_id", "UNKNOWN")

            if rule_id in excluded:
                continue

            active_violations.append(v)

        return active_violations

    def should_fail(self, violations: List[Dict[str, Any]]) -> bool:
        """
        Determine if the process should exit with error.
        """
        if not violations:
            return False

        if self.config.get("strict_mode"):
            return True

        # Fail if any error severity
        return any(v.get("severity") == "error" for v in violations)
