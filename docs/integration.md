# Integration Guide

Saguaro is designed to be the "Cortex" for other AI Agents. This guide explains how to integrate Saguaro into your agentic workflow.

## 1. CLI Usage (JSON Mode)
All Saguaro commands support structured output, making them easy to parse.

### Retrieval
```bash
saguaro query "database connection pool" --json
```

**Output:**
```json
{
  "matches": [
    {
      "file": "src/db/pool.py",
      "score": 0.89,
      "type": "class",
      "line": 45
    }
  ]
}
```

### Perception (SSAI)
Use the **Standard Agent Interface (SSAI)** to read code.

1.  **Get Context**: `saguaro agent skeleton path/to/file.py`
2.  **Read Function**: `saguaro agent slice ClassName.function`

## 2. Model Context Protocol (MCP)
Saguaro exposes an MCP-compatible server.

```bash
saguaro serve
```
This starts a stdio or SSE server that exposing resources:
*   `saguaro://query?q=...`
*   `saguaro://file/...`

## 3. Python API
For deep integration, import the `saguaro` package directly.

```python
from saguaro.indexing.engine import IndexEngine

# Load existing index
engine = IndexEngine(repo_path=".")

# Vector Search
results = engine.search("authentication", k=5)

for res in results:
    print(f"{res.file}: {res.score}")
```
