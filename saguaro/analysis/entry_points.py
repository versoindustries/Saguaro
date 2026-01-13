
"""
Entry Point Detector
Identifies runtime entry points like CLI commands, API routes, and main blocks.
"""

import os
import ast
from typing import List, Dict, Any

class EntryPointDetector:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        
    def detect(self) -> List[Dict[str, Any]]:
        entry_points = []
        for root, _, files in os.walk(self.root_dir):
            if '.git' in root or 'venv' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    entry_points.extend(self._scan_file(path))
        return entry_points

    def _scan_file(self, path: str) -> List[Dict]:
        found = []
        try:
            with open(path, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # 1. Main block
                if isinstance(node, ast.If):
                    if (isinstance(node.test, ast.Compare) and 
                        isinstance(node.test.left, ast.Name) and 
                        node.test.left.id == "__name__" and 
                        isinstance(node.test.comparators[0], ast.Constant) and 
                        node.test.comparators[0].value == "__main__"):
                        found.append({
                            "type": "main_block",
                            "file": path,
                            "line": node.lineno
                        })
                
                # 2. Flask/FastAPI Routes (Decorator heuristics)
                if isinstance(node, ast.FunctionDef):
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Call):
                            # @app.route() or @router.get()
                            if isinstance(dec.func, ast.Attribute):
                                if dec.func.attr in ['route', 'get', 'post', 'put', 'delete']:
                                    found.append({
                                        "type": "api_route",
                                        "file": path,
                                        "line": node.lineno,
                                        "name": node.name
                                    })
                            
                # 3. Click/Argparse (Heuristic)
                # Checking for click.command()
                if isinstance(node, ast.FunctionDef):
                    for dec in node.decorator_list:
                         if isinstance(dec, ast.Attribute) and dec.attr == 'command':
                             found.append({"type": "cli_command", "file": path, "line": node.lineno, "name": node.name})
                         elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == 'command':
                              found.append({"type": "cli_command", "file": path, "line": node.lineno, "name": node.name})

        except Exception:
            pass
        return found
