
import os
import json
import logging
import uuid
import shutil
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)

class Sandbox:
    """
    In-memory Shadow Filesystem for safe Agent operations.
    Allows applying patches, verifying them, and committing only when safe.
    """
    
    _instances = {}

    def __init__(self, repo_path: str, created_at: float = None, sandbox_id: str = None):
        self.id = sandbox_id or str(uuid.uuid4())[:8]
        self.repo_path = os.path.abspath(repo_path)
        self.created_at = created_at or time.time()
        
        self.storage_dir = os.path.join(self.repo_path, ".saguaro", "sandboxes", self.id)
        
        # Shadow state: path -> content (string)
        self.shadow_files: Dict[str, str] = {}
        self.patches_applied: List[Dict] = []
        
        # Try load
        if sandbox_id and os.path.exists(self.storage_dir):
            self._load()
        else:
            Sandbox._instances[self.id] = self

    @classmethod
    def get(cls, sandbox_id: str, repo_path: str = None) -> Optional['Sandbox']:
        # Check memory first
        if sandbox_id in cls._instances:
            return cls._instances[sandbox_id]
        
        # Check disk
        repo = repo_path or os.getcwd()
        s_dir = os.path.join(repo, ".saguaro", "sandboxes", sandbox_id)
        if os.path.exists(s_dir):
            return cls(repo, sandbox_id=sandbox_id)
            
        return None

    def _save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        # Save manifest
        manifest = {
            "id": self.id,
            "created_at": self.created_at,
            "patches": self.patches_applied
        }
        with open(os.path.join(self.storage_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f)
            
        # Save shadow files
        files_dir = os.path.join(self.storage_dir, "files")
        os.makedirs(files_dir, exist_ok=True)
        
        for path, content in self.shadow_files.items():
            # hash path to safe filename
            import hashlib
            safe_name = hashlib.md5(path.encode()).hexdigest()
            # map in manifest?
            # simplified: assume we iterate shadow_files
            with open(os.path.join(files_dir, safe_name), 'w') as f:
                f.write(content)
            # Store mapping
            with open(os.path.join(files_dir, safe_name + ".meta"), 'w') as f:
                f.write(path)

    def _load(self):
        with open(os.path.join(self.storage_dir, "manifest.json"), 'r') as f:
            data = json.load(f)
            self.created_at = data['created_at']
            self.patches_applied = data['patches']
            
        files_dir = os.path.join(self.storage_dir, "files")
        if os.path.exists(files_dir):
            for fname in os.listdir(files_dir):
                if fname.endswith(".meta"): continue
                
                # content
                with open(os.path.join(files_dir, fname), 'r') as f:
                    content = f.read()
                
                # path
                meta_file = os.path.join(files_dir, fname + ".meta")
                if os.path.exists(meta_file):
                    with open(meta_file, 'r') as f:
                        path = f.read().strip()
                    self.shadow_files[path] = content

    def read_file(self, file_path: str) -> str:
        """Reads from shadow if modified, else from disk."""
        abs_path = os.path.abspath(file_path) if not os.path.isabs(file_path) else file_path
        
        if abs_path in self.shadow_files:
            return self.shadow_files[abs_path]
        
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        raise FileNotFoundError(f"File not found: {abs_path}")

    def apply_patch(self, patch: Dict[str, Any]):
        """
        Applies a semantic patch to the shadow state.
        Patch format: { "target_file": "...", "operations": [...] }
        """
        target_file = patch.get("target_file")
        if not target_file:
            raise ValueError("Patch missing target_file")
            
        abs_path = os.path.join(self.repo_path, target_file)
        
        # Read content via self.read_file (accounts for previous shadow writes)
        try:
             content = self.read_file(abs_path)
        except Exception:
             content = "" # New file?
        
        # Naive patching
        for op in patch.get("operations", []):
            op_type = op.get("op")
            
            if op_type == "replace" or op_type == "insert":
                # For this prototype, we assume 'content' is the WHOLE new file content
                # if op="overwrite", or we handle simple replacement if logic matches?
                # The roadmap spec says "Semantic Patch".
                # For simplicity in this demo, let's treat "content" as new file content if overwrite
                # or just doing a string replace if symbol provided.
                
                if op_type == "replace" and op.get("symbol") is None:
                     # Full file overwrite
                     content = op.get("content")
                else:
                     # naive
                     content = op.get("content")
                     
        self.shadow_files[abs_path] = content
        self.patches_applied.append(patch)
        self._save() # Persist!
        return self.id

    def verify(self) -> Dict[str, Any]:
        """
        Runs logic verification on the shadow state.
        """
        report = {"status": "pass", "violations": []}
        
        for path, content in self.shadow_files.items():
            # Check syntax (Python)
            if path.endswith(".py"):
                try:
                    compile(content, path, 'exec')
                except SyntaxError as e:
                    report["status"] = "fail"
                    report["violations"].append(f"SyntaxError in {os.path.basename(path)}: {e}")

        return report

    def calculate_impact(self) -> Dict[str, Any]:
        """Runs impact analysis on modified files."""
        from saguaro.analysis.impact import ImpactAnalyzer
        from saguaro.simulation.volatility import VolatilityMapper
        
        analyzer = ImpactAnalyzer(self.repo_path)
        vol_mapper = VolatilityMapper()
        
        # Volatility Map (Expensive? Cache it?)
        # For prototype, generate on fly or mock
        vmap = vol_mapper.generate_map(self.repo_path)
        
        impact_report = {
            "sandbox_id": self.id,
            "risk_score": 0.0,
            "files": []
        }
        
        max_risk = 0.0
        
        for path in self.shadow_files:
            rel_path = os.path.relpath(path, self.repo_path)
            
            # Downstream Impact
            analysis = analyzer.analyze_change(path)
            
            # Volatility Risk
            volatility = vmap.get(rel_path, 0.0)
            
            # Combined Risk Score (Heuristic)
            # Impact dependents * Volatility
            risk = len(analysis["tests_impacted"] + analysis["interfaces_impacted"]) * (1.0 + volatility)
            
            if risk > max_risk: max_risk = risk
            
            impact_report["files"].append({
                "path": rel_path,
                "dependents": len(analysis["interfaces_impacted"]),
                "tests": len(analysis["tests_impacted"]),
                "volatility": volatility,
                "risk": risk
            })
            
        impact_report["risk_score"] = max_risk
        return impact_report

    def commit(self):
        """Flushes shadow state to disk and triggers micro-indexing."""
        modified_paths = []
        for path, content in self.shadow_files.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            modified_paths.append(path)
        
        # Micro-Indexing (Drift-Aware)
        try:
             from saguaro.indexing.engine import IndexEngine
             from saguaro.indexing.auto_scaler import get_repo_stats_and_config
             
             logger.info(f"Micro-indexing {len(modified_paths)} files...")
             saguaro_dir = os.path.join(self.repo_path, ".saguaro")
             stats = get_repo_stats_and_config(self.repo_path)
             engine = IndexEngine(self.repo_path, saguaro_dir, stats)
             engine.index_batch(modified_paths, force=True) # Force re-index specifically these
             engine.commit()
        except Exception as e:
             logger.error(f"Micro-indexing failed: {e}")
             # Don't fail the commit itself?
        
        # Clear persist
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
            
        applied = len(self.shadow_files)
        self.shadow_files = {}
        return applied
