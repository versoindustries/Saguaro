
import os
import shutil
import logging
from typing import Dict, Any
from saguaro.analysis.impact import ImpactAnalyzer
from saguaro.analysis.dead_code import DeadCodeAnalyzer

logger = logging.getLogger(__name__)

class SafetyEngine:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.impact_analyzer = ImpactAnalyzer(self.repo_path)
        self.dead_code = DeadCodeAnalyzer(self.repo_path)

    def safe_delete(self, target_path: str, force: bool = False, dry_run: bool = True) -> Dict[str, Any]:
        """
        Deletes a file if and only if it has zero inbound dependencies (or forced).
        """
        abs_path = os.path.abspath(target_path)
        if not os.path.exists(abs_path):
             return {"success": False, "message": "File not found"}

        # 1. Check Impact
        try:
             report = self.impact_analyzer.analyze_change(abs_path)
        except Exception as e:
             logger.error(f"Impact analysis failed: {e}")
             if not force:
                  return {"success": False, "message": "Impact analysis failed", "error": str(e)}
             report = {"impact_score": 0} # Force assume 0 if forced

        dependents = []
        if 'tests_impacted' in report: dependents.extend(report['tests_impacted'])
        if 'interfaces_impacted' in report: dependents.extend(report['interfaces_impacted'])
        
        # Self-references don't count
        dependents = [d for d in dependents if os.path.abspath(d) != abs_path]

        if dependents and not force:
             return {
                 "success": False, 
                 "message": "Unsafe to delete: Inbound dependencies detected.",
                 "blocking_dependents": dependents
             }
             
        # 2. Heuristics check (False Positives)
        # reflection, config, entry points
        # Placeholder for heuristic check
        
        # 3. Execution
        if dry_run:
             return {"success": True, "message": "Safe to delete (Dry Run)", "impact_score": 0}
        
        # Move to trash or delete?
        # For enterprise safety, move to .saguaro/trash/<timestamp>
        trash_dir = os.path.join(self.repo_path, ".saguaro", "trash")
        os.makedirs(trash_dir, exist_ok=True)
        import time
        dest = os.path.join(trash_dir, f"{os.path.basename(abs_path)}_{int(time.time())}")
        
        try:
            shutil.move(abs_path, dest)
            return {"success": True, "message": f"Moved to trash: {dest}"}
        except Exception as e:
            return {"success": False, "message": f"Deletion failed: {e}"}
