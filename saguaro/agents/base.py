from abc import ABC, abstractmethod
from typing import Dict, Any
from saguaro.context import Context
from saguaro.client import SAGUAROClient


class Agent(ABC):
    """Base class for SAGUARO specialized agents."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.client = SAGUAROClient()  # Access to SAGUARO Core

    @abstractmethod
    def run(self, context: Context, **kwargs) -> Dict[str, Any]:
        """Execute the agent's main logic."""
        pass

    def log_activity(self, message: str):
        print(f"[{self.role.upper()}] {self.name}: {message}")
