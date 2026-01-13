"""
SAGUARO Rule Engine
Defines compliance rules and logic to load them.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import yaml
import os
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class Rule:
    id: str
    pattern: str
    message: str
    severity: str = "ERROR" # ERROR, WARN
    scope: str = "**" # Glob pattern
    replacement: Optional[str] = None
    
    def check(self, content: str) -> List[Tuple[int, str]]:
        """
        Checks content for violations.
        Returns list of (line_context, match_string) tuples.
        """
        violations = []
        try:
            regex = re.compile(self.pattern)
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if regex.search(line):
                    violations.append((i+1, line.strip()))
        except re.error as e:
            logger.error(f"Invalid regex for rule {self.id}: {e}")
            
        return violations

class RuleLoader:
    @staticmethod
    def load(repo_path: str) -> List[Rule]:
        rules_path = os.path.join(repo_path, ".saguaro.rules")
        if not os.path.exists(rules_path):
            logger.info("No .saguaro.rules found.")
            return []
            
        try:
            with open(rules_path, 'r') as f:
                data = yaml.safe_load(f)
                
            rules = []
            for item in data.get("rules", []):
                rules.append(Rule(
                    id=item.get("id", "unknown"),
                    pattern=item.get("pattern"),
                    message=item.get("message"),
                    severity=item.get("severity", "ERROR"),
                    scope=item.get("scope", "**"),
                    replacement=item.get("replacement")
                ))
            return rules
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return []
