
import os
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class ContextAnchor:
    """Defines the precise location of a snippet within a file."""
    start_line: int
    end_line: int
    entity_name: str
    entity_type: str

@dataclass
class ContextItem:
    """A single unit of context (atomic)."""
    id: str  # unique identifier (e.g. hash)
    content: str
    file_path: str
    score: float
    anchor: ContextAnchor
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ContextBundle:
    """
    A deterministic, ordered collection of context items.
    """
    query_text: str
    timestamp: float
    items: List[ContextItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    workset_id: str = None
    constraints: List[str] = field(default_factory=list)
    scope_files: List[str] = field(default_factory=list)
    scope_symbols: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
        
    def add_item(self, item: ContextItem):
        self.items.append(item)
        
    def sort(self):
        """
        Enforce deterministic ordering:
        1. Score (descending)
        2. File path (lexicographical)
        3. Start line (ascending)
        """
        self.items.sort(key=lambda x: (-x.score, x.file_path, x.anchor.start_line))

class ContextBuilder:
    """Helper to construct bundles from raw search results."""
    
    @staticmethod
    def build_from_results(query_text: str, results: List[Dict[str, Any]], timestamp: float, workset: Any = None) -> ContextBundle:
        ws_id = workset.id if workset else None
        
        # Serialize constraints if present
        constraints = []
        scope_files = []
        scope_symbols = []
        
        if workset:
            if hasattr(workset, 'constraints'):
                 for c in workset.constraints:
                      constraints.append(str(c))
            if hasattr(workset, 'files'):
                 scope_files = sorted(workset.files)
            if hasattr(workset, 'symbols'):
                 scope_symbols = sorted(workset.symbols)

        bundle = ContextBundle(
            query_text=query_text, 
            timestamp=timestamp, 
            workset_id=ws_id,
            constraints=constraints,
            scope_files=scope_files,
            scope_symbols=scope_symbols
        )
        
        for res in results:
            # Construct Anchor
            anchor = ContextAnchor(
                start_line=res.get('line', 0),
                end_line=res.get('end_line', 0), # Assuming end_line usually available or 0
                entity_name=res.get('name', 'unknown'),
                entity_type=res.get('type', 'unknown')
            )
            
            # Construct Item
            # Create a localized ID
            item_id = f"{res.get('file')}:{res.get('line')}"
            
            content = res.get('content', '')
            file_path = res.get('file', '')
            
            # If content missing, try to fetch from disk
            if not content and file_path and os.path.exists(file_path):
                try:
                    start = int(res.get('line', 1))
                    end = int(res.get('end_line', start + 10))
                    with open(file_path, 'r', errors='ignore') as f:
                        lines = f.readlines()
                        # Adjust for 0-indexing
                        start_idx = max(0, start - 1)
                        end_idx = min(len(lines), end)
                        content = "".join(lines[start_idx:end_idx])
                except Exception as e:
                    content = f"<error reading content: {e}>"

            item = ContextItem(
                id=item_id,
                content=content or "<content unavailable>", 
                file_path=file_path,
                score=res.get('score', 0.0),
                anchor=anchor,
                reason=res.get('reason', '')
            )
            
            bundle.add_item(item)
            
        bundle.sort()
        bundle.metadata["count"] = len(bundle.items)
        return bundle

Context = ContextBundle
