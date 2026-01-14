import ast
import os
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class SkeletonGenerator:
    """Generates a Skeleton view of code files."""

    def generate(self, file_path: str) -> Dict[str, Any]:
        """Generates a skeleton dictionary for the given file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path = os.path.abspath(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        skeleton = {
            "type": "skeleton",
            "file_path": file_path,
            "language": self._detect_language(file_path),
            "loc": len(content.splitlines()),
            "symbols": [],
            "imports": [],
        }

        if skeleton["language"] == "python":
            self._parse_python(content, skeleton)
        else:
            # Fallback or Todo for other languages
            skeleton["note"] = (
                "Skeleton generation only supported for Python in this version."
            )

        return skeleton

    def _detect_language(self, path: str) -> str:
        if path.endswith(".py"):
            return "python"
        if path.endswith(".js"):
            return "javascript"
        if path.endswith(".ts"):
            return "typescript"
        if path.endswith((".c", ".h", ".cpp", ".cc")):
            return "cpp"
        return "unknown"

    def _parse_python(self, content: str, skeleton: Dict[str, Any]):
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error parsing {skeleton['file_path']}: {e}")
            skeleton["error"] = str(e)
            return

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    skeleton["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    skeleton["imports"].append(f"{module}.{alias.name}")

        # Extract symbols (Top level)
        for node in tree.body:
            symbol = self._visit_node(node)
            if symbol:
                skeleton["symbols"].append(symbol)

    def _visit_node(self, node: ast.AST) -> Dict[str, Any] | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._visit_function(node)
        elif isinstance(node, ast.ClassDef):
            return self._visit_class(node)
        return None

    def _visit_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        docstring = ast.get_docstring(node)

        # Reconstruct signature (simplified)
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    # ast.unparse available in 3.9+
                    if hasattr(ast, "unparse"):
                        ann = ast.unparse(arg.annotation)
                        arg_str += f": {ann}"
                    else:
                        arg_str += ": <type>"
                except Exception:
                    pass
            args.append(arg_str)

        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        sig = f"{prefix} {node.name}({', '.join(args)})"

        if node.returns:
            try:
                if hasattr(ast, "unparse"):
                    ret = ast.unparse(node.returns)
                    sig += f" -> {ret}"
                else:
                    sig += " -> <type>"
            except Exception:
                pass

        return {
            "name": node.name,
            "type": "function",
            "signature": sig,
            "line_start": node.lineno,
            "line_end": node.end_lineno,
            "docstring": docstring,
        }

    def _visit_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        docstring = ast.get_docstring(node)
        children = []
        for item in node.body:
            child = self._visit_node(item)
            if child:
                if child["type"] == "function":
                    child["type"] = "method"
                children.append(child)

        return {
            "name": node.name,
            "type": "class",
            "line_start": node.lineno,
            "line_end": node.end_lineno,
            "docstring": docstring,
            "children": children,
        }


class SliceGenerator:
    """Generates a contextual slice of the codebase."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.metadata_path = os.path.join(
            repo_path, ".saguaro", "vectors", "metadata.json"
        )
        self.metadata = []
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    def generate(self, symbol_name: str, depth: int = 1) -> Dict[str, Any]:
        matches = [m for m in self.metadata if m.get("name") == symbol_name]

        if not matches:
            if "." in symbol_name:
                parts = symbol_name.split(".")
                c_matches = [m for m in self.metadata if m.get("name") == parts[-1]]
                if c_matches:
                    matches = c_matches

        if not matches:
            # Return actionable error response (Phase 2: AI Adoption)
            # This guides AI models to use the correct Saguaro workflow
            is_file_path = "/" in symbol_name or symbol_name.endswith(".py")

            if is_file_path:
                suggestion = f"""This looks like a file path. For file exploration, use:
  saguaro agent skeleton {symbol_name}
For semantic search, use:
  saguaro query 'description of code you need' --k 5"""
            else:
                suggestion = f"""Symbol '{symbol_name}' was not found in the index.
This could mean:
  1. The symbol name is incorrect (check spelling/case)
  2. The index is stale (run: saguaro index --path .)
  3. The symbol doesn't exist in the codebase

Try these alternatives:
  - Semantic search: saguaro query "{symbol_name}" --k 5
  - List file symbols: saguaro agent skeleton <file_path>
  - Rebuild index: saguaro index --path ."""

            return {
                "error": "Symbol not found",
                "type": "INDEX_MISS",
                "symbol": symbol_name,
                "suggestion": suggestion,
                "fallback_allowed": True,
                "recovery_steps": [
                    f'saguaro query "{symbol_name}" --k 5',
                    "saguaro health",
                    "saguaro index --path .",
                ],
            }

        target = matches[0]

        result = {
            "type": "slice",
            "focus_symbol": symbol_name,
            "depth": depth,
            "content": [],
        }

        focus_code = self._get_code(target)
        result["content"].append(
            {
                "role": "focus",
                "name": target.get("name"),
                "file": target.get("file"),
                "type": target.get("type"),
                "code": focus_code,
            }
        )

        imports = self._get_imports(target.get("file"))
        for imp in imports:
            result["content"].append(
                {
                    "role": "dependency",
                    "relation": "import",
                    "name": imp,
                    "file": target.get("file"),
                    "signature": f"import {imp}",
                }
            )

        return result

    def _get_code(self, meta: Dict[str, Any]) -> str:
        path = meta.get("file")
        start = meta.get("line", 1)
        end = meta.get("end_line", 10000)

        full_path = path if os.path.isabs(path) else os.path.join(self.repo_path, path)

        if not os.path.exists(full_path):
            return "<File not found>"

        with open(full_path, "r") as f:
            lines = f.readlines()

        return "".join(lines[start - 1 : end])

    def _get_imports(self, file_path: str) -> List[str]:
        full_path = (
            file_path
            if os.path.isabs(file_path)
            else os.path.join(self.repo_path, file_path)
        )
        if not os.path.exists(full_path) or not full_path.endswith(".py"):
            return []

        try:
            with open(full_path, "r") as f:
                tree = ast.parse(f.read())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module if node.module else ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            return imports
        except Exception:
            return []
