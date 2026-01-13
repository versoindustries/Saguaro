"""
SAGUARO Parsing Module
Wraps tree-sitter to extract semantically enhanced entities from source code.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)

try:
    import importlib.util
    TREE_SITTER_AVAILABLE = importlib.util.find_spec("tree_sitter") is not None
    if TREE_SITTER_AVAILABLE:
        from tree_sitter import Language, Parser # noqa: F401
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter not installed. Parsing will fallback to basic text split.")

class CodeEntity:
    def __init__(self, name: str, type: str, content: str, start_line: int, end_line: int, file_path: str):
        self.name = name
        self.type = type
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.file_path = file_path
        
    def __repr__(self):
        return f"<CodeEntity {self.name} ({self.type})>"

class SAGUAROParser:
    def __init__(self):
        self.languages = {}
        if TREE_SITTER_AVAILABLE:
            try:
                from tree_sitter_languages import get_language, get_parser
                self.get_language = get_language
                self.get_parser = get_parser
            except ImportError:
                logger.warning("tree_sitter_languages not found. Fallback to basic.")
                self.get_language = None

    def parse_file(self, file_path: str) -> List[CodeEntity]:
        """
        Parses a file and returns a list of CodeEntities.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        entities = []
        
        # Try Tree-sitter first
        if TREE_SITTER_AVAILABLE and self.get_language:
            try:
                lang_name = None
                if file_path.endswith('.py'):
                    lang_name = 'python'
                elif file_path.endswith('.cc') or file_path.endswith('.cpp'):
                    lang_name = 'cpp'
                elif file_path.endswith('.c'):
                    lang_name = 'c'
                elif file_path.endswith('.js'):
                    lang_name = 'javascript'
                elif file_path.endswith('.ts'):
                    lang_name = 'typescript'
                
                if lang_name:
                    parser = self.get_parser(lang_name)
                    tree = parser.parse(bytes(content, "utf8"))
                    
                    # Simple query for definitions
                    if lang_name == 'python':
                        query = self.get_language('python').query("""
                        (function_definition name: (identifier) @func.name) @func.def
                        (class_definition name: (identifier) @class.name) @class.def
                        """)
                        captures = query.captures(tree.root_node)
                        # Process captures to build entities
                        # Note: tree-sitter API changes frequently, this is a simplified view
                        processed_nodes = set()
                        for node, tag in captures:
                            if node.id in processed_nodes:
                                continue
                            
                            if tag.endswith('.def'):
                                type_ = 'class' if 'class' in tag else 'function'
                                # Find name node (approximate if query didn't perfectly align pairs)
                                name = "unknown"
                                for child in node.children:
                                    if child.type == 'identifier':
                                        name = content[child.start_byte:child.end_byte]
                                        break
                                        
                                entities.append(CodeEntity(
                                    name=name,
                                    type=type_,
                                    content=content[node.start_byte:node.end_byte],
                                    start_line=node.start_point[0] + 1,
                                    end_line=node.end_point[0] + 1,
                                    file_path=file_path
                                ))
                                processed_nodes.add(node.id)

            except Exception as e:
                logger.debug(f"Tree-sitter parse failed for {file_path}: {e}")

        # If entities found via TS, return them + file entity
        if entities:
            entities.append(CodeEntity(
                name=os.path.basename(file_path),
                type="file",
                content=content,
                start_line=1,
                end_line=content.count('\n')+1,
                file_path=file_path
            ))
            return entities

        # Fallback / Simple Parser for prototype (if TS failed or file type not supported)
        lines = content.splitlines()
        
        # Simple heuristic parser for Python
        if file_path.endswith(".py"):
            for i, line in enumerate(lines):
                strip = line.strip()
                if strip.startswith("def ") or strip.startswith("class "):
                    name = strip.split(" ")[1].split("(")[0].split(":")[0]
                    type_ = "class" if strip.startswith("class ") else "function"
                    end = min(len(lines), i + 20) 
                    block = "\n".join(lines[i:end])
                    
                    entities.append(CodeEntity(
                        name=name,
                        type=type_,
                        content=block,
                        start_line=i+1,
                        end_line=end,
                        file_path=file_path
                    ))
        
        # Always add the file itself as a "module" entity
        entities.append(CodeEntity(
            name=os.path.basename(file_path),
            type="file",
            content=content,
            start_line=1,
            end_line=len(lines),
            file_path=file_path
        ))
        
        return entities
