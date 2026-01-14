import os
import json
import logging
from typing import Optional
from .objects import Plan

logger = logging.getLogger(__name__)


class RoadmapManager:
    def __init__(self, saguaro_dir: str):
        self.plans_dir = os.path.join(saguaro_dir, "plans")
        os.makedirs(self.plans_dir, exist_ok=True)

    def save_plan(self, plan: Plan):
        import dataclasses

        path = os.path.join(self.plans_dir, f"{plan.id}.json")
        with open(path, "w") as f:
            # Simple serialization
            f.write(json.dumps(dataclasses.asdict(plan), indent=2))

    def load_plan(self, plan_id: str) -> Optional[Plan]:
        path = os.path.join(self.plans_dir, f"{plan_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            json.load(f)
            # Basic reconstruction logic needed
            # For prototype, we skip deep obj reconstruction
            return None  # Todo: Implement full hydration

    def update_roadmap_state(self):
        """
        Scans Plans and checks if tasks are completed (via git or verify).
        """
        pass  # Placeholder for dynamic invalidation logic
