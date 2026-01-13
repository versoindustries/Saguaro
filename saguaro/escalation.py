
import os
import re
import logging
from typing import List, Dict, Any
from saguaro.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DependencyResolver:
    """
    Simple heuristic resolver for finding immediate dependencies.
    """
    def __init__(self, root_path: str):
        self.root_path = root_path

    def get_imports(self, file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            return []
            
        imports = set()
        
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
            # Python Import Logic
            if file_path.endswith('.py'):
                # from x.y import z -> x/y.py
                # import x.y -> x/y.py
                
                # Regex for 'from module import ...'
                from_matches = re.finditer(r'^from\s+([\w\.]+)\s+import', content, re.MULTILINE)
                for m in from_matches:
                    module = m.group(1)
                    path = self._resolve_python_module(module)
                    if path:
                        imports.add(path)
                    
                # Regex for 'import module'
                import_matches = re.finditer(r'^import\s+([\w\.]+)', content, re.MULTILINE)
                for m in import_matches:
                    module = m.group(1)
                    path = self._resolve_python_module(module)
                    if path:
                        imports.add(path)

            # C++ Include Logic
            elif file_path.endswith(('.cc', '.cpp', '.h', '.hpp')):
                # #include "foo/bar.h"
                include_matches = re.finditer(r'#include\s+["<]([\w\./]+)[">]', content)
                for m in include_matches:
                    path = self._resolve_cpp_include(m.group(1))
                    if path:
                        imports.add(path)
                    
        except Exception as e:
            logger.warning(f"Error resolving imports for {file_path}: {e}")
            
        return list(imports)

    def _resolve_python_module(self, module: str) -> str:
        # Convert dots to slashes
        rel_path = module.replace('.', '/')
        
        # Check standard locations
        # 1. Local file
        candidate = os.path.join(self.root_path, rel_path + ".py")
        if os.path.exists(candidate):
            return candidate
        
        # 2. Package dir
        candidate_init = os.path.join(self.root_path, rel_path, "__init__.py")
        if os.path.exists(candidate_init):
            return candidate_init
        
        # 3. Top-level heuristic (if module is saguaro.x, check x)
        parts = rel_path.split('/')
        if len(parts) > 1:
             # Try resolving from root based on sub-parts
             # e.g. saguaro/core/x -> saguaro/core/x.py
             sub_path = os.path.join(self.root_path, *parts)
             if os.path.exists(sub_path + ".py"):
                 return sub_path + ".py"
             if os.path.exists(os.path.join(sub_path, "__init__.py")):
                 return os.path.join(sub_path, "__init__.py")
             
        return None

    def _resolve_cpp_include(self, include_path: str) -> str:
        candidate = os.path.join(self.root_path, include_path)
        if os.path.exists(candidate):
            return candidate
        return None

class EscalationLadder:
    def __init__(self, vector_store: VectorStore, root_path: str):
        self.store = vector_store
        self.root_path = root_path
        self.resolver = DependencyResolver(root_path)

    def search(self, query_vec: Any, seed_file: str, level: int = 3, k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes a search at the specified escalation level.
        
        Level 0: Local (Seed File Only)
        Level 1: 1-Hop (Seed + Immediate Imports)
        Level 2: 2-Hop (Import of Imports - simplistic)
        Level 3: Global (Full Index)
        """
        allow_list = None # None means all allowed
        
        if level < 3 and seed_file:
            seed_file = os.path.abspath(seed_file)
            allow_list = {seed_file}
            
            if level >= 1:
                imports = self.resolver.get_imports(seed_file)
                allow_list.update(imports)
                
            if level >= 2:
                # Expand one more level
                secondary_imports = set()
                for imp in list(allow_list):
                    secondary_imports.update(self.resolver.get_imports(imp))
                allow_list.update(secondary_imports)
        
        # Now run query on VectorStore, but filter results based on allow_list
        # VectorStore.query is naive, so we get more results (k*factor) and filter, 
        # or we hack VectorStore to accept a filter.
        # Since we can't change C++ ops easily, we'll fetch more from store and filter in Python.
        
        # If Level 3, just query normal
        if allow_list is None:
            return self.store.query(query_vec, k=k)
            
        # If filtering, we need to fetch enough candidates to satisfy k after filtering.
        # This is a limitation of the current naive store.
        # Strategy: Fetch 10*k, filter, if not enough, sad.
        raw_results = self.store.query(query_vec, k=k*20) 
        
        filtered = []
        for res in raw_results:
            res_path = res.get('file')
            if res_path in allow_list:
                filtered.append(res)
        
        # Add explanation of scope
        for res in filtered:
            if res.get('file') == seed_file:
                 res['scope'] = 'Local (Seed)'
            else:
                 res['scope'] = f'Level {level} Import'

        return filtered[:k]

