
"""
Dead Code Analyzer
Identifies unreachable code and calculates confidence scores for safe deletion.
"""

import os
import ast
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DeadCodeAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.definitions = {} # name -> file
        self.references = set() # name
        
    def analyze(self) -> List[Dict]:
        """
        Scans codebase for definitions and references to find unused code.
        Returns list of dead code candidates with confidence scores.
        """
        self._scan_repo()
        
        candidates = []
        for name, file_path in self.definitions.items():
            if name not in self.references:
                # Potential dead code
                # Exclude common entry points or magic methods
                if self._is_ignored(name):
                    continue
                    
                score = self._calculate_confidence(name, file_path)
                candidates.append({
                    "symbol": name,
                    "file": file_path,
                    "confidence": score,
                    "reason": "No static references found"
                })
                
        # Sort by confidence
        return sorted(candidates, key=lambda x: x['confidence'], reverse=True)

    def _scan_repo(self):
        """Walks repo and populates definitions and references."""
        for root, dirs, files in os.walk(self.repo_path):
            if '.git' in dirs:
                dirs.remove('.git')
            if 'venv' in dirs:
                dirs.remove('venv')
            if '.saguaro' in dirs:
                dirs.remove('.saguaro')
            
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    self._parse_file(path)

    def _parse_file(self, path: str):
        try:
            with open(path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                # Definitions
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'): # Ignore private for now
                        self.definitions[node.name] = path
                elif isinstance(node, ast.ClassDef):
                    self.definitions[node.name] = path
                
                # References
                if isinstance(node, ast.Name):
                    self.references.add(node.id)
                elif isinstance(node, ast.Attribute):
                    self.references.add(node.attr)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.references.add(node.func.id)
        except Exception:
            pass

    def _is_ignored(self, name: str) -> bool:
        if name.startswith('__') and name.endswith('__'):
            return True
        if name in ['main', 'setup', 'teardown']:
            return True
        return False

    def _calculate_confidence(self, name: str, file_path: str) -> float:
        """
        Calculates a confidence score (0.0 - 1.0) that the code is truly dead.
        Factors:
        - Is it in a 'tests' directory? (Low risk usually, but maybe not 'dead' just unused helper)
        - Is it a common framework name? (e.g. Django views)
        """
        score = 0.8 # Base confidence
        
        if 'tests' in file_path:
            score -= 0.2 # Might be a helper used by implicit discovery
            
        if 'views.py' in file_path or 'urls.py' in file_path:
            score -= 0.5 # Web frameworks often use string references or routing
            
        return max(0.0, min(1.0, score))

