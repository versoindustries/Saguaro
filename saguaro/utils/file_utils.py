
import os
import fnmatch
from typing import List

def get_code_files(root_path: str, exclusions: List[str] = None) -> List[str]:
    """
    Scans the directory for code and configuration files, applying exclusions.
    Supports a broad range of extensions relevant to modern full-stack development.
    """
    if exclusions is None:
        exclusions = []
        
    # Ensure hard defaults are always present
    hard_defaults = ['.saguaro', '.git', 'venv', '__pycache__', 'node_modules', 'build', 'dist', '.idea', '.vscode']
    exclusions.extend([d for d in hard_defaults if d not in exclusions])
    
    # Supported Extensions
    # We use a set of whitelisted extensions to avoid indexing binaries/media
    EXT_WHITELIST = {
        # Python
        '.py', '.pyi',
        # C/C++
        '.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx',
        # Web
        '.js', '.mjs', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', '.less',
        # Backend/Systems
        '.java', '.go', '.rs', '.rb', '.php', '.cs', '.kt', '.swift',
        # Shell
        '.sh', '.bash', '.zsh',
        # Config / Data
        '.json', '.yaml', '.yml', '.toml', '.xml', 'Dockerfile', 'Makefile', '.ini',
        # Docs
        '.md', '.rst', '.txt'
    }
    
    all_files = []
    
    for root, dirs, files in os.walk(root_path):
        # 1. Prune directories (modify dirs in-place)
        dirs_to_remove = []
        for d in dirs:
            # Check exact match
            if d in exclusions:
                dirs_to_remove.append(d)
                continue
            # Check ignored prefix (e.g. .hidden)
            if d.startswith('.'):
                 dirs_to_remove.append(d)
                 continue
            # Check globs
            for pat in exclusions:
                if fnmatch.fnmatch(d, pat):
                    dirs_to_remove.append(d)
                    break
                    
        for d in dirs_to_remove:
            if d in dirs:
                dirs.remove(d)
                
        # 2. Filter files
        for file in files:
            # Check file exclusions
            skip_file = False
            for pat in exclusions:
                 if fnmatch.fnmatch(file, pat):
                     skip_file = True
                     break
            if skip_file:
                continue
                
            # Extension check
            _, ext = os.path.splitext(file)
            
            # Special case for "extensionless" files like Dockerfile or Makefile
            if file in EXT_WHITELIST: # Exact filename match
                full_path = os.path.join(root, file)
                all_files.append(full_path)
                continue
                
            if ext in EXT_WHITELIST:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
                
    return all_files
