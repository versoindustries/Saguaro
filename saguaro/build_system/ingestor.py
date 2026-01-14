"""
Build Graph Ingestor
Parses build configuration files to construct a dependency graph of the project structure.
Supports: Python (setup.py/pyproject.toml), Node (package.json), C++ (CMake/Make).
"""

import os
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class BuildTarget:
    name: str  # e.g. "saguaro", "test_core"
    type: str  # "lib", "bin", "test", "external"
    file: str  # Path to definition file
    dependencies: List[str] = field(
        default_factory=list
    )  # List of target names or packages
    sources: List[str] = field(default_factory=list)  # Source files included


class BuildGraphIngestor:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.targets: Dict[str, BuildTarget] = {}

    def ingest(self) -> Dict[str, Any]:
        """Scans repository and builds the graph."""
        self._scan_python()
        self._scan_node()
        self._scan_cmake()
        self._scan_makefile()

        return {
            "root": self.root_dir,
            "target_count": len(self.targets),
            "targets": {k: self._target_to_dict(v) for k, v in self.targets.items()},
        }

    def _target_to_dict(self, t: BuildTarget) -> Dict:
        return {
            "type": t.type,
            "file": os.path.relpath(t.file, self.root_dir),
            "deps": t.dependencies,
            "sources": [os.path.relpath(s, self.root_dir) for s in t.sources],
        }

    def _scan_python(self):
        # Scan for setup.py
        for root, dirs, files in os.walk(self.root_dir):
            if "setup.py" in files:
                path = os.path.join(root, "setup.py")
                self._parse_setup_py(path)
            if "pyproject.toml" in files:
                path = os.path.join(root, "pyproject.toml")
                # Simplifying: just treating as a target marker
                name = os.path.basename(root) or "root"
                self.targets[f"py:{name}"] = BuildTarget(
                    name=name, type="lib", file=path
                )

    def _parse_setup_py(self, path: str):
        try:
            with open(path, "r") as f:
                content = f.read()

            # Simple regex to find name and install_requires
            name_match = re.search(r'name=["\']([^"\']+)["\']', content)
            name = name_match.group(1) if name_match else "unknown_python"

            deps = []
            # Very loose regex for requirements
            reqs_match = re.findall(r'[\'"]([a-zA-Z0-9_\-]+)[<>=]', content)
            deps.extend(reqs_match)

            self.targets[f"py:{name}"] = BuildTarget(
                name=name,
                type="lib",
                file=path,
                dependencies=deps,
                sources=[path],  # And presumably the package dir, ignored for now
            )
        except Exception:
            pass

    def _scan_node(self):
        for root, _, files in os.walk(self.root_dir):
            if "package.json" in files:
                path = os.path.join(root, "package.json")
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        name = data.get("name", "unknown_node")
                        deps = list(data.get("dependencies", {}).keys())
                        dev_deps = list(data.get("devDependencies", {}).keys())

                        self.targets[f"npm:{name}"] = BuildTarget(
                            name=name,
                            type="lib",
                            file=path,
                            dependencies=deps + dev_deps,
                        )
                except Exception:
                    pass

    def _scan_cmake(self):
        for root, _, files in os.walk(self.root_dir):
            if "CMakeLists.txt" in files:
                path = os.path.join(root, "CMakeLists.txt")
                self._parse_cmake(path)

    def _parse_cmake(self, path: str):
        try:
            with open(path, "r") as f:
                content = f.read()

            # Find add_library or add_executable
            # add_library(name source1 source2 ...)
            libs = re.findall(
                r"add_library\s*\(\s*(\w+)\s+([^)]+)\)", content, re.DOTALL
            )
            for lib_name, sources_str in libs:
                sources = sources_str.split()
                # filter sources
                sources = [os.path.join(os.path.dirname(path), s) for s in sources]
                self.targets[f"cmake:{lib_name}"] = BuildTarget(
                    name=lib_name, type="lib", file=path, sources=sources
                )

            exes = re.findall(
                r"add_executable\s*\(\s*(\w+)\s+([^)]+)\)", content, re.DOTALL
            )
            for exe_name, sources_str in exes:
                sources = sources_str.split()
                sources = [os.path.join(os.path.dirname(path), s) for s in sources]
                self.targets[f"cmake:{exe_name}"] = BuildTarget(
                    name=exe_name, type="bin", file=path, sources=sources
                )
        except Exception:
            pass

    def _scan_makefile(self):
        # Very heuristic
        pass
