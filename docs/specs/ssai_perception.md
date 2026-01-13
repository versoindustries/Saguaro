# SSAI Perception Specs: Skeleton & Slice

This document defines the schema for the "Perception" layer of the SAGUARO Standard Agent Interface (SSAI). 
These structures allow agents to "see" the codebase in a token-efficient manner.

## 1. The Skeleton (File View)

A **Skeleton** is a high-level summary of a source file. It strips away implementation details (function bodies) while retaining structural and semantic information (signatures, types, docstrings).

### Schema (JSON)

```json
{
  "type": "skeleton",
  "file_path": "src/core/engine.py",
  "language": "python",
  "loc": 150,
  "symbols": [
    {
      "name": "Engine",
      "type": "class",
      "line_start": 10,
      "line_end": 150,
      "docstring": "Core semantic processing engine.",
      "children": [
        {
          "name": "process",
          "type": "method",
          "signature": "def process(self, input: str) -> dict",
          "line_start": 20,
          "line_end": 45,
          "docstring": "Processes the input string and returns a metadata dictionary."
        }
      ]
    }
  ],
  "imports": ["typing", "os"]
}
```

### Goals
- **Token Efficiency**: Eliminate 80%+ of tokens (logic) while keeping 100% of the API surface.
- **Navigability**: Agents can scan directories of skeletons to locate relevant files before requesting full content.

---

## 2. The Slice (Context View)

A **Slice** is a dependency-aware graph of code centered around a specific "Focus Symbol". It includes the symbol's full source code and the signatures/docs of its immediate dependencies (callees, types).

### Schema (JSON)

```json
{
  "type": "slice",
  "focus_symbol": "Engine.process",
  "depth": 1,
  "content": [
    {
      "role": "focus",
      "name": "Engine.process",
      "file": "src/core/engine.py",
      "code": "    def process(self, input: str) -> dict:\n        # ... full implementation ...\n        return metadata"
    },
    {
      "role": "dependency",
      "relation": "calls",
      "name": "Metadata.create",
      "file": "src/core/types.py",
      "signature": "def create(name: str) -> dict",
      "docstring": "Factory for metadata objects."
    }
  ],
  "tokens": 450
}
```

### Goals
- **Precision**: Provide exactly the code needed to understand or modify the focus symbol.
- **Hallucination Reduction**: Explicitly explicitly include dependency signatures so the agent doesn't guess them.
- **Self-Containment**: A slice should be "compile-ready" or close to it, for isolated reasoning.
