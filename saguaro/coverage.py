import os
import logging
from typing import Dict
from saguaro.parsing.parser import SAGUAROParser

logger = logging.getLogger(__name__)

class CoverageReporter:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.parser = SAGUAROParser()
        self.ext_map = {
            # Core & Systems
            '.py': 'Python',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.cpp': 'C++', '.cc': 'C++', '.cxx': 'C++', '.hpp': 'C++ Header', '.hh': 'C++ Header',
            '.rs': 'Rust',
            '.go': 'Go',
            '.java': 'Java',
            '.cs': 'C#',
            '.swift': 'Swift',
            '.kt': 'Kotlin', '.kts': 'Kotlin',
            '.scala': 'Scala',
            
            # Web
            '.js': 'JavaScript', '.jsx': 'JavaScript', '.mjs': 'JavaScript',
            '.ts': 'TypeScript', '.tsx': 'TypeScript',
            '.html': 'HTML', '.htm': 'HTML',
            '.css': 'CSS', '.scss': 'Sass', '.sass': 'Sass', '.less': 'Less',
            '.php': 'PHP',
            '.vue': 'Vue',
            '.svelte': 'Svelte',
            '.dart': 'Dart',
            
            # Scripting
            '.rb': 'Ruby',
            '.pl': 'Perl', '.pm': 'Perl',
            '.lua': 'Lua',
            '.sh': 'Shell', '.bash': 'Shell', '.zsh': 'Shell',
            '.ps1': 'PowerShell',
            
            # Data/ML
            '.r': 'R',
            '.jl': 'Julia',
            '.sql': 'SQL',
            
            # Functional
            '.hs': 'Haskell',
            '.erl': 'Erlang',
            '.ex': 'Elixir', '.exs': 'Elixir',
            '.clj': 'Clojure',
            '.lisp': 'Lisp', '.lsp': 'Lisp',
            '.ml': 'OCaml',
            '.fs': 'F#',
            
            # Config/Data Formats
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yaml': 'YAML', '.yml': 'YAML',
            '.toml': 'TOML',
            '.xml': 'XML',
            '.ini': 'INI',
            '.conf': 'Config',
            '.env': 'Config',
            '.dockerfile': 'Docker',
            '.cmake': 'CMake',
            
            # Legacy/Other
            '.f': 'Fortran', '.f90': 'Fortran',
            '.cob': 'COBOL',
            '.vb': 'Visual Basic',
            '.pas': 'Pascal',
            '.ada': 'Ada',
            '.sol': 'Solidity',
            '.graphql': 'GraphQL',
            '.proto': 'Protocol Buffer',
            '.txt': 'Text',
            '.in': 'Text',
        }
        self.filename_map = {
            'CMakeLists.txt': 'CMake',
            'Dockerfile': 'Docker',
            'Makefile': 'Make',
            'Gnumakefile': 'Make',
            'Jenkinsfile': 'Groovy',
            'Vagrantfile': 'Ruby',
            'Rakefile': 'Ruby',
            'Gemfile': 'Ruby Gemfile',
            'Procfile': 'Procfile',
            'requirements.txt': 'Pip Requirements',
            'Pipfile': 'Pipfile',
            'pyproject.toml': 'Python Project',
            'cargo.toml': 'Rust Crate',
            'package.json': 'NPM Package',
            'go.mod': 'Go Module',
            'go.sum': 'Go Sum',
            'LICENSE': 'Text',
            'NOTICE': 'Text',
            'AUTHORS': 'Text',
            'OWNERS': 'Text',
            '.gitignore': 'Git Ignore',
            '.dockerignore': 'Docker Ignore',
        }
        self.excludes = {
            '.git', '.venv', 'venv', 'node_modules', '__pycache__', 'build', 'dist', 
            '.idea', '.vscode', '.saguaro',
            '.ruff_cache', '.pytest_cache', '.mypy_cache', '.git_cache'
        }

    def generate_report(self) -> Dict:
        stats = {
            "total_files": 0,
            "languages": {},
            "ast_supported_files": 0,
            "blind_files": 0,
            "blind_list": []
        }
        
        # Check parser capabilities
        ts_available = False
        try:
            import importlib.util
            ts_available = importlib.util.find_spec("tree_sitter") is not None
        except ImportError:
            pass

        # Supported languages by our parser wrapper (based on simple code reading of parser.py)
        # In parser.py: python, cpp, c, javascript, typescript are explicitly handled with get_parser
        parser_supported = {'Python', 'C++', 'C', 'JavaScript', 'TypeScript'}

        for root, dirs, files in os.walk(self.root_path):
            # Prune directories
            # We copy the list to iterate and modify the original 'dirs' list in place to affect os.walk recursion
            active_dirs = list(dirs)
            dirs[:] = []
            
            for d in active_dirs:
                if d in self.excludes:
                    continue
                if d.endswith('.egg-info'):
                    continue
                dirs.append(d)
            
            for file in files:
                ext = os.path.splitext(file)[1]
                full_path = os.path.join(root, file)
                
                lang = None
                if file in self.filename_map:
                    lang = self.filename_map[file]
                elif ext in self.ext_map:
                    lang = self.ext_map[ext]
                
                if lang:
                    stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                    stats["total_files"] += 1
                    
                    if lang in parser_supported and ts_available:
                        stats["ast_supported_files"] += 1
                    else:
                        # Even if language is known, if we don't have AST parser, it's "text only"
                        # For now, let's consider logic files without parser as "partial coverage"
                        # But strictly per "AST coverage", it's 0.
                        pass
                        
                else:
                    # Unsupported / Blind spots
                    # filter out binary blobs or irrelevant files if we want, 
                    # but for now count anything unknown as blind.
                    # Maybe skip obvious lock files or configs? 
                    # Keeping it simple.
                    stats["blind_files"] += 1
                    if len(stats["blind_list"]) < 20: # Cap list
                        stats["blind_list"].append(os.path.relpath(full_path, self.root_path))
                        
        return stats

    def print_report(self):
        stats = self.generate_report()
        total = stats["total_files"]
        ast = stats["ast_supported_files"]
        
        print("\n=== SAGUARO Coverage Report ===")
        print(f"Total Tracked Files: {total}")
        
        print("\n[Languages Detected]")
        for lang, count in sorted(stats["languages"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count}")
            
        print("\n[AST Coverage]")
        if total > 0:
            coverage = (ast / total) * 100
            print(f"  Coverage: {coverage:.1f}% ({ast}/{total} files)")
        else:
            print("  Coverage: N/A")
            
        print("\n[Blind Spots (Unsupported Files)]")
        print(f"  Total Unknown: {stats['blind_files']}")
        if stats["blind_list"]:
            print("  Sample:")
            for f in stats["blind_list"]:
                print(f"    - {f}")
        if stats["blind_files"] > 20:
            print(f"    ... and {stats['blind_files'] - 20} more.")
            
        print("\n===============================")
