from dataclasses import dataclass, field
from typing import List
import time
import uuid


@dataclass
class Task:
    id: str
    description: str
    target_files: List[str]
    type: str = "refactor"  # refactor, cleanup, migration
    status: str = "pending"  # pending, in_progress, complete, failed
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    risk_score: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class Plan:
    id: str
    goal: str
    scope: str
    tasks: List[Task] = field(default_factory=list)
    risk_summary: str = "Unknown"

    def add_task(
        self, description: str, targets: List[str], type: str = "refactor"
    ) -> Task:
        t = Task(
            id=str(uuid.uuid4())[:8],
            description=description,
            target_files=targets,
            type=type,
        )
        self.tasks.append(t)
        return t
