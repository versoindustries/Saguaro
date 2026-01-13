
import logging
from .objects import Plan
from saguaro.refactor.planner import RefactorPlanner

logger = logging.getLogger(__name__)

class TaskDecompositionEngine:
    def __init__(self, repo_path: str):
        self.refactor_planner = RefactorPlanner(repo_path)

    def decompose_goal(self, goal: str, target_symbol: str = None) -> Plan:
        """
        Decomposes a high-level goal into a Plan with Tasks.
        Example: goal="Rename User to Customer"
        """
        plan_id = f"plan_{int(time.time())}"
        plan = Plan(id=plan_id, goal=goal, scope="global") # simplification

        if target_symbol:
            # Generate refactor plan
            refactor_plan = self.refactor_planner.plan_symbol_modification(target_symbol)
            
            # create tasks from phases
            for i, p in enumerate(refactor_plan.get('phases', [])): # Assuming phases list of dicts
                 # p = {'file': '...', 'order': 1}
                 task = plan.add_task(
                     description=f"Refactor usage in {p['file']}",
                     targets=[p['file']],
                     type="refactor"
                 )
                 # previous task dependency
                 if i > 0:
                     task.dependencies.append(plan.tasks[i-1].id)
                     
            plan.risk_summary = f"Impact Score: {refactor_plan['impact_score']}"

        return plan
        
import time
