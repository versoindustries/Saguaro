
"""
Refactor Planner
Generates execution plans for refactoring tasks by analyzing dependencies and impact.
"""

import os
import ast
import logging
from typing import List, Dict, Any
from saguaro.indexing.engine import IndexEngine

logger = logging.getLogger(__name__)

class DependencyGraph:
    def __init__(self):
        self.edges = {} # file -> list of imported files
        self.reverse_edges = {} # file -> list of importers

    def add_dependency(self, source: str, target: str):
        if source not in self.edges:
            self.edges[source] = set()
        self.edges[source].add(target)
        
        if target not in self.reverse_edges:
            self.reverse_edges[target] = set()
        self.reverse_edges[target].add(source)

class RefactorPlanner:
    def __init__(self, repo_path: str, engine: IndexEngine = None):
        self.repo_path = os.path.abspath(repo_path)
        self.engine = engine
        
    def plan_symbol_modification(self, symbol_name: str) -> Dict[str, Any]:
        """
        Analyze the impact of modifying a symbol (e.g. rename/change signature).
        """
        # 1. Find candidates via Index (or global grep if no index)
        candidates = self._find_candidates(symbol_name)
        
        # 2. Verify usage via AST
        verified_usages = self._verify_usages(symbol_name, candidates)
        
        # 3. Build Dependency Graph of impacted files
        # We need the graph of ALL impacted files to sort them.
        graph = self._build_dependency_graph(list(verified_usages.keys()))
        
        # 4. Analyze API Risk
        api_risk = self._analyze_api_risk(symbol_name, verified_usages, candidates)
        
        # 5. Generate Phased Plan
        ordered_files = self._schedule_changes(graph, verified_usages)
        
        plan = {
            "symbol": symbol_name,
            "impact_score": len(verified_usages),
            "api_surface_risk": api_risk,
            "files_impacted": ordered_files,
            "modules": self._group_by_module(verified_usages.keys()),
            "phases": [{"order": i+1, "file": f, "reason": "Dependency chain"} for i, f in enumerate(ordered_files)]
        }
        
        return plan

    def _find_candidates(self, symbol: str) -> List[str]:
        # Fast text search
        matches = []
        for root, _, files in os.walk(self.repo_path):
            if '.git' in root or 'venv' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', errors='ignore') as f:
                            if symbol in f.read():
                                matches.append(path)
                    except Exception:
                        pass
        return matches

    def _verify_usages(self, symbol: str, files: List[str]) -> Dict[str, List[int]]:
        """
        Parse files to confirm symbol is used as an identifier.
        """
        usage_map = {}
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                lines = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == symbol:
                        lines.append(node.lineno)
                    elif isinstance(node, ast.FunctionDef) and node.name == symbol:
                        lines.append(node.lineno)
                    elif isinstance(node, ast.ClassDef) and node.name == symbol:
                        lines.append(node.lineno)
                    elif isinstance(node, ast.Attribute) and node.attr == symbol:
                        lines.append(node.lineno)
                
                if lines:
                    usage_map[file_path] = sorted(list(set(lines)))
            except Exception:
                pass
                
        return usage_map

    def _analyze_api_risk(self, symbol: str, usages: Dict[str, List[int]], files: List[str]) -> str:
        """Determines if the symbol is part of the public API surface."""
        risk = "Low (Internal)"
        
        # 1. Check naming convention
        if not symbol.startswith('_'):
            risk = "Medium (Public Name)"
            
        # 2. Check if exported in __init__.py
        for f in files:
            if f.endswith('__init__.py') and f in usages:
                risk = "High (Exported in __init__)"
                break
                
        # 3. Check for external usage (outside own module)
        modules = self._group_by_module(usages.keys())
        if len(modules) > 2:
             risk = f"{risk} - Highly Coupled ({len(modules)} modules)"
             
        return risk

    def _build_dependency_graph(self, files: List[str]) -> DependencyGraph:
        graph = DependencyGraph()
        # Simplified import parsing for prototype
        # Real implementation would resolve imports to file paths
        return graph

    def _group_by_module(self, files):
        groups = {}
        for f in files:
            rel = os.path.relpath(f, self.repo_path)
            parts = rel.split(os.sep)
            top = parts[0] if len(parts) > 1 else 'root'
            if top not in groups:
                groups[top] = []
            groups[top].append(rel)
        return groups

    def _schedule_changes(self, graph, verified_usages):
        """
        Topological sort of files based on dependencies.
        """
        # Since _build_dependency_graph is a stub for now, we just return list
        # In future: implement proper topo sort here
        return list(verified_usages.keys())
