from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEngine(ABC):
    """
    Abstract base class for analysis engines.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.policy_config = {}

    def set_policy(self, config: Dict[str, Any]):
        self.policy_config = config

    @abstractmethod
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the engine and return a list of violations.
        Each violation is a dict with:
        - file: str
        - line: int
        - rule_id: str
        - message: str
        - severity: str ('error', 'warning', etc)
        - context: str (optional)
        """
        pass

    def fix(self, violation: Dict[str, Any]) -> bool:
        """
        Attempt to fix a specific violation.
        Returns True if fixed, False otherwise.
        """
        return False
