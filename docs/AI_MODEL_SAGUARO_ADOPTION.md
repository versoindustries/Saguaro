# AI Model Saguaro Adoption: Problem Analysis & Solutions

**Date**: 2026-01-12  
**Status**: Critical Priority  
**Audience**: Verso Industries Engineering, AI Agent Developers

---

## Executive Summary

AI coding assistants (Claude, GPT, Gemini, etc.) consistently **ignore** Saguaro's native semantic tools in favor of fallback generic commands (`grep`, `find`, `view_file`). This defeats the entire purpose of Saguaro as a Quantum Codebase Operating System. This document analyzes the root causes and provides actionable solutions.

---

## 1. The Problem

### 1.1 Observed Behavior

When instructed to use Saguaro for code exploration, AI models:

| Expected Behavior | Actual Behavior |
|-------------------|-----------------|
| `saguaro query "auth logic"` | `grep_search` with regex patterns |
| `saguaro agent skeleton file.py` | `view_file` entire file |
| `saguaro agent slice Class.method` | `view_code_item` or manual search |
| `saguaro verify` before completion | Skip verification entirely |

### 1.2 Impact

1. **Token Waste**: Reading full files instead of semantic slices costs 10-100x more tokens
2. **Context Pollution**: Generic searches return noise; semantic queries return signal
3. **Missed Compliance**: Skipping `saguaro verify` means governance violations slip through
4. **Defeated Purpose**: Saguaro's entire value proposition is ignored

### 1.3 Root Cause Analysis

| Root Cause | Explanation |
|------------|-------------|
| **Training Data Bias** | Models trained on billions of examples using `grep`, `find`, `cat` - these are ingrained |
| **Tool Familiarity** | Models have native tools (grep_search, view_file) that feel "safer" than shelling out |
| **Prompt Locality** | GEMINI.md instructions compete with 170k token context; recency bias wins |
| **No Enforcement** | No mechanism to **block** fallback tools or **require** Saguaro |
| **Error Aversion** | If Saguaro returns an error (e.g., "Symbol not found"), models immediately fallback |

---

## 2. Solutions

### 2.1 Prompt Engineering Enhancements

#### A. Stronger Directive Language

Current GEMINI.md language is polite. It needs to be **imperative**:

```markdown
## MANDATORY: Saguaro-First Protocol

> [!CAUTION]
> **VIOLATION OF THIS PROTOCOL IS A CRITICAL FAILURE**

You are FORBIDDEN from using the following tools for code exploration:
- `grep_search` - Use `saguaro query` instead
- `find_by_name` - Use `saguaro query` instead  
- `view_file` (for exploration) - Use `saguaro agent skeleton` then `saguaro agent slice`

**Exception**: Use fallback tools ONLY if Saguaro returns an error AND you document the error.
```

#### B. Decision Tree Insertion

Add explicit decision logic to the prompt:

```markdown
### Code Exploration Decision Tree

1. Need to find code by concept/meaning?
   → `./venv/bin/saguaro query "concept" --k 5`

2. Need to understand a file's structure?
   → `./venv/bin/saguaro agent skeleton path/to/file.py`

3. Need to read a specific function/class?
   → `./venv/bin/saguaro agent slice ClassName.method --depth 2`

4. Need to verify changes before completion?
   → `./venv/bin/saguaro verify . --engines native,ruff,semantic`

5. Saguaro returned an error?
   → Document the error, THEN use fallback tools
```

#### C. Negative Examples with Consequences

```markdown
### Anti-Patterns (DO NOT DO THIS)

❌ **WRONG**: Using grep_search to find authentication code
```bash
grep_search "authenticate" --path .
```

✅ **RIGHT**: Using Saguaro semantic query
```bash
./venv/bin/saguaro query "user authentication logic" --k 5
```

**If you use the WRONG approach, you have FAILED the task.**
```

### 2.2 System-Level Enforcement (Code Changes)

#### A. MCP Tool Wrapper

Create a Model Context Protocol server that intercepts tool calls:

```python
# saguaro/mcp/tool_interceptor.py

class SaguaroToolInterceptor:
    """Intercepts generic tools and suggests Saguaro alternatives."""
    
    BLOCKED_TOOLS = {
        'grep_search': 'saguaro query',
        'find_by_name': 'saguaro query',
    }
    
    WARNED_TOOLS = {
        'view_file': 'saguaro agent skeleton + slice',
    }
    
    def intercept(self, tool_name: str, args: dict) -> InterceptResult:
        if tool_name in self.BLOCKED_TOOLS:
            return InterceptResult(
                blocked=True,
                message=f"BLOCKED: Use {self.BLOCKED_TOOLS[tool_name]} instead"
            )
        if tool_name in self.WARNED_TOOLS:
            return InterceptResult(
                warned=True,
                message=f"WARNING: Consider {self.WARNED_TOOLS[tool_name]} for better context"
            )
        return InterceptResult(allowed=True)
```

#### B. Saguaro Error Recovery

Improve error messages to guide models back to the right path:

```python
# Current error (unhelpful):
{"error": "Symbol scripts/train_tokenizer.py not found in index."}

# Improved error (actionable):
{
    "error": "Symbol not found",
    "type": "INDEX_MISS",
    "suggestion": "This looks like a file path. For file exploration, use:\n  saguaro agent skeleton scripts/train_tokenizer.py\nFor semantic search, use:\n  saguaro query 'tokenizer training' --k 5",
    "fallback_allowed": true
}
```

#### C. Auto-Reindex Hook

Reduce "symbol not found" errors by auto-indexing on first query:

```python
def query(self, text: str, k: int = 5):
    results = self._search(text, k)
    if not results and not self._index_verified:
        self._trigger_index()
        results = self._search(text, k)
    return results
```

### 2.3 Agent Framework Integration

For AI agent frameworks (LangChain, AutoGPT, CrewAI), provide native tool definitions:

```python
# saguaro/integrations/langchain_tools.py

from langchain.tools import BaseTool

class SaguaroQueryTool(BaseTool):
    name = "saguaro_semantic_search"
    description = """
    PRIMARY tool for finding code by meaning/concept.
    Use this INSTEAD of grep or file searches.
    Input: Natural language description of what you're looking for
    Output: Ranked list of relevant code entities with file paths
    """
    
    def _run(self, query: str) -> str:
        result = subprocess.run(
            ["./venv/bin/saguaro", "query", query, "--k", "5", "--json"],
            capture_output=True, text=True
        )
        return result.stdout

class SaguaroSkeletonTool(BaseTool):
    name = "saguaro_file_skeleton"
    description = """
    Use this INSTEAD of viewing full files.
    Shows function/class signatures and docstrings only.
    Saves 90% of tokens compared to full file reads.
    """
    # ...
```

### 2.4 Metrics & Observability

Track adoption to measure improvement:

```yaml
# .saguaro/metrics.yaml

tool_usage:
  saguaro_query: 0
  saguaro_skeleton: 0
  saguaro_slice: 0
  saguaro_verify: 0
  
fallback_usage:
  grep_search: 0
  find_by_name: 0
  view_file: 0

adoption_score: 0.0  # saguaro / (saguaro + fallback)
```

```bash
# CLI command to view adoption metrics
./venv/bin/saguaro metrics --session
```

---

## 3. Implementation Roadmap

| Phase | Task | Priority | Status | Implementation |
|-------|------|----------|--------|----------------|
| **1** | Update GEMINI.md with stronger directives | Critical | ✅ Done | Added mandatory protocol, decision tree, anti-patterns |
| **2** | Improve error messages in Saguaro CLI | High | ✅ Done | `perception.py` returns actionable errors with recovery steps |
| **3** | Add decision tree to agent skeleton/slice help | High | ✅ Done | Enhanced `--help` text in CLI with usage guidance |
| **4** | Create MCP tool interceptor | Medium | ✅ Done | `saguaro/mcp/tool_interceptor.py` |
| **5** | Add LangChain/CrewAI tool wrappers | Medium | ✅ Done | `saguaro/integrations/langchain_tools.py` |
| **6** | Implement adoption metrics | Low | ✅ Done | `saguaro/mcp/adoption_metrics.py` + `saguaro metrics` CLI |

---

## 4. Updated GEMINI.md Section (Ready to Merge)

```markdown
## Native Agent Protocol (SOP) -- "Saguaro First"

> [!CAUTION]
> **MANDATORY PROTOCOL - VIOLATION IS TASK FAILURE**

### Tool Hierarchy (MUST FOLLOW)

| Task | Required Tool | FORBIDDEN Alternatives |
|------|---------------|----------------------|
| Find code by concept | `saguaro query "..."` | grep_search, find_by_name |
| Explore file structure | `saguaro agent skeleton file.py` | view_file (full read) |
| Read function/class | `saguaro agent slice Entity.method` | view_code_item |
| Verify before commit | `saguaro verify .` | (skipping verification) |

### Error Handling

If Saguaro returns an error:
1. **Document the error** in your response
2. **Check if index exists**: `./venv/bin/saguaro health`
3. **Rebuild if needed**: `./venv/bin/saguaro index --path .`
4. **Only then** use fallback tools, with explicit justification

### Examples

**Finding authentication logic:**
```bash
# ✅ CORRECT
./venv/bin/saguaro query "user authentication and session management" --k 5

# ❌ WRONG - DO NOT USE
grep_search "auth" --path .
```

**Exploring a file:**
```bash
# ✅ CORRECT - Step 1: Get skeleton
./venv/bin/saguaro agent skeleton saguaro/core.py

# ✅ CORRECT - Step 2: Read specific entity
./venv/bin/saguaro agent slice CoreEngine.process --depth 2

# ❌ WRONG - Wastes tokens
view_file saguaro/core.py  # Reading 500+ lines
```
```

---

## 5. Conclusion

The adoption failure is **not a Saguaro problem** - it's a **prompt engineering and enforcement problem**. AI models will use whatever tools are most familiar and least error-prone. To achieve Saguaro-first behavior:

1. **Strengthen directives** with explicit prohibitions and consequences
2. **Improve error messages** to guide models back on track
3. **Add enforcement mechanisms** at the MCP/tool layer
4. **Measure adoption** and iterate

The goal is to make Saguaro the **path of least resistance**, not just the "preferred" option.

---

*Document Author: Verso Industries Engineering*  
*Last Updated: 2026-01-12*
