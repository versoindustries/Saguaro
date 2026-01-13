
import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

from saguaro.governor import ContextGovernor, ContextBudgetExceeded

logger = logging.getLogger(__name__)

@dataclass
class WorksetConstraint:
    type: str  # "read_only", "no_new_deps", "tests_must_pass"
    target: str # e.g. "saguaro/core/**"

@dataclass
class Workset:
    id: str
    description: str
    files: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    constraints: List[WorksetConstraint] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    status: str = "active" # active, locked, closed
    budget_usage: int = 0
    budget_limit: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(json_str: str) -> 'Workset':
        data = json.loads(json_str)
        constraints = [WorksetConstraint(**c) for c in data.get("constraints", [])]
        data["constraints"] = constraints
        return Workset(**data)

class WorksetManager:
    def __init__(self, saguaro_dir: str, repo_path: str = "."):
        self.saguaro_dir = saguaro_dir
        self.repo_path = os.path.abspath(repo_path)
        self.worksets_dir = os.path.join(saguaro_dir, "worksets")
        os.makedirs(self.worksets_dir, exist_ok=True)
        self.governor = ContextGovernor()

    def _estimate_file_tokens(self, rel_path: str) -> int:
        full_path = os.path.join(self.repo_path, rel_path)
        try:
            if not os.path.exists(full_path):
                return 0
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return self.governor.estimate_tokens(content)
        except Exception as e:
            logger.warning(f"Could not read {rel_path} for token estimation: {e}")
            return 0

    def create_workset(self, description: str, files: List[str], symbols: List[str] = None, constraints: List[dict] = None, allow_escalation: bool = False) -> Workset:
        """Creates a new workset with budget enforcement."""
        
        # 1. Budget Check
        total_tokens = 0
        file_items = []
        for f in files:
            tokens = self._estimate_file_tokens(f)
            total_tokens += tokens
            file_items.append({"name": f, "content": " " * (tokens * 4)}) # Mock content for governor check

        # Check soft limit first
        is_safe, est, msg = self.governor.check_budget(file_items)
        
        if not is_safe and not allow_escalation:
            # If not safe (hard limit exceeded), we reject
            raise ContextBudgetExceeded(f"Workset creation failed: {msg}")

        # If warning (soft limit) but not hard limit, we allow it but log/warn
        if "WARNING" in msg:
            logger.warning(f"Workset near budget limit: {msg}")

        # Generate ID based on content hash + time for uniqueness
        content = f"{description}{sorted(files)}{time.time()}"
        ws_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        
        real_constraints = []
        if constraints:
            for c in constraints:
                real_constraints.append(WorksetConstraint(**c))

        ws = Workset(
            id=ws_id,
            description=description,
            files=files,
            symbols=symbols or [],
            constraints=real_constraints,
            status="pending", # Must acquire lease explicitly
            budget_usage=total_tokens,
            budget_limit=self.governor.hard_limit
        )
        self._save(ws)
        return ws

    def expand_workset(self, ws_id: str, new_files: List[str], justification: str) -> Workset:
        """Attempts to add files to an existing workset."""
        ws = self.get_workset(ws_id)
        if not ws:
            raise ValueError("Workset not found")

        # Calculate new usage
        current_files = set(ws.files)
        added_tokens = 0
        for f in new_files:
            if f not in current_files:
                added_tokens += self._estimate_file_tokens(f)
        
        new_usage = ws.budget_usage + added_tokens
        
        if new_usage > ws.budget_limit:
            # Check justification? For now, we just enforce the hard limit unless specific key words used?
            # Or maybe we allow escalation if justification is provided (mock logic)
            if "CRITICAL" in justification.upper() or "SECURITY" in justification.upper():
                 # Allow bump
                 new_limit = self.governor.escalate(ws.budget_limit)
                 ws.budget_limit = new_limit
                 if new_usage > new_limit:
                      raise ContextBudgetExceeded(f"Even with escalation, limit exceeded ({new_usage} > {new_limit})")
            else:
                 raise ContextBudgetExceeded(f"Expansion rejected. Budget {ws.budget_usage} -> {new_usage} exceeds limit {ws.budget_limit}. Justification insufficient.")

        # Apply expansion
        for f in new_files:
            if f not in ws.files:
                ws.files.append(f)
        
        ws.budget_usage = new_usage
        self._save(ws)
        return ws

    def get_workset(self, ws_id: str) -> Optional[Workset]:
        path = os.path.join(self.worksets_dir, f"{ws_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return Workset.from_json(f.read())

    def list_worksets(self) -> List[Workset]:
        worksets = []
        if not os.path.exists(self.worksets_dir):
            return []
            
        for f_name in os.listdir(self.worksets_dir):
            if f_name.endswith(".json"):
                with open(os.path.join(self.worksets_dir, f_name), "r") as f:
                    try:
                        worksets.append(Workset.from_json(f.read()))
                    except Exception:
                        continue # Skip malformed
        return sorted(worksets, key=lambda w: w.created_at, reverse=True)

    def _save(self, workset: Workset):
        path = os.path.join(self.worksets_dir, f"{workset.id}.json")
        with open(path, "w") as f:
            f.write(workset.to_json())

    def check_conflicts(self, proposed_files: List[str], exclude_ws_id: str = None) -> List[Dict]:
        """
        Checks if any proposed files are already claimed by other ACTIVE worksets.
        """
        conflicts = []
        active = [w for w in self.list_worksets() if w.status == "active" and w.id != exclude_ws_id]
        
        proposed_set = set(proposed_files)
        
        for ws in active:
            existing_set = set(ws.files)
            overlap = proposed_set.intersection(existing_set)
            if overlap:
                conflicts.append({
                    "workset_id": ws.id,
                    "description": ws.description,
                    "overlapping_files": list(overlap)
                })
        return conflicts

    def acquire_lease(self, ws_id: str) -> Dict[str, Any]:
        """
        Attempts to lock the workset (set status to active) if no conflicts exist.
        """
        ws = self.get_workset(ws_id)
        if not ws:
             return {"success": False, "message": "Workset not found"}
             
        conflicts = self.check_conflicts(ws.files, exclude_ws_id=ws_id)
        if conflicts:
            return {
                "success": False, 
                "message": "Conflicts detected", 
                "conflicts": conflicts
            }
            
        ws.status = "active"
        self._save(ws)
        return {"success": True, "message": "Lease acquired", "workset": ws}

    def release_lease(self, ws_id: str):
        """Releases the lease (sets status to closed)."""
        ws = self.get_workset(ws_id)
        if ws:
            ws.status = "closed"
            self._save(ws)
