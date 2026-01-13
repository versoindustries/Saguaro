# Saguaro Native Agent Protocol (SNAP)

## Overview
This document defines the standard protocol for AI Models and Agents interacting with the Saguaro Quantum Codebase OS. By following this protocol, agents reduce token usage, increase semantic accuracy, and ensure compliance with governance rules.

## Core Principle: "Saguaro First"
Agents must prioritize **Native Saguaro Tools** over generic filesystem tools (`view_file`, `grep`, `ls`) whenever possible. Saguaro tools provide dependency-aware, token-optimized views of the codebase.

---

## 1. Discovery & Navigation
**Goal**: Find relevant code without blindly reading files.

| Generic Tool | **Saguaro Native Tool** | Why? |
| :--- | :--- | :--- |
| `grep_search` | `saguaro query "concept" --k 5` | Finds code by *meaning* (semantic), not just text match. |
| `find_by_name` | `saguaro query "filename" --k 1` | Retrieves the file's canonical location and context context. |
| `list_dir` | `saguaro agent skeleton <file>` | Lists symbols (functions/classes) instead of raw files, providing immediate structure. |

### Protocol
1.  **Do not guess paths.** Start with `saguaro query`.
2.  **Do not `ls -R`.** Use the index to find where logic lives.

---

## 2. Perception (Reading Code)
**Goal**: Read code to understand logic or prepare for edits.

| Generic Tool | **Saguaro Native Tool** | Why? |
| :--- | :--- | :--- |
| `view_file` (Full) | `saguaro agent skeleton <file>` | **Low Token Cost**. See the shape of the file first (signatures + docstrings). |
| `view_file` (Range) | `saguaro agent slice <symbol> --depth 1` | **High Context**. See the function *plus* its imports and dependencies automatically. |
| `read_file` | `saguaro query "..." --json` | Returns code *with* metadata in a structured format. |

### Protocol
1.  **Skeleton First**: Before reading a whole file, run `skeleton` to key landmarks.
2.  **Slice for Logic**: When analyzing a bug or feature, pull a `slice` of the relevant function/class. This brings in necessary context (imports, parent class) that `view_file` at line 10-20 might miss.

---

## 3. Action (Editing Code)
**Goal**: Modify code safely.

| Generic Tool | **Saguaro Native Tool** | Why? |
| :--- | :--- | :--- |
| `replace_file_content` | `saguaro agent patch <file> <json>` | Applies changes transactionally via Sandbox. |
| `run_test` | `saguaro agent verify <sandbox_id>` | Verifies *only* the impacted blast radius, not the whole suite. |

### Protocol
1.  **Sandbox**: All edits should ideally happen in a Saguaro Sandbox (if enabled).
2.  **Verify**: Never commit without running `saguaro verify` (Sentinel).

---

## 5. Refactoring (Semantic Safety)
**Goal**: Rename or move symbols without breaking references across the codebase.

| Generic Tool | **Saguaro Native Tool** | Why? |
| :--- | :--- | :--- |
| `grep` + `replace` | `saguaro refactor rename <old> <new>` | **Safe**. Uses the index to find actual references, avoiding false positives (e.g. matching a string literal instead of a variable). |
| Mental Model | `saguaro refactor plan --symbol <name>` | **Predictive**. Tells you exactly which files *will* break before you touch them. |

### Protocol
1.  **Plan First**: Run `refactor plan` before any major change.
2.  **Execute via Tool**: Use `refactor rename` for renames instead of manual multi-file edits.

---

## 6. Impact Analysis (Pre-Edit)
**Goal**: Understand the blast radius of a change to minimize regression risk.

| Generic Tool | **Saguaro Native Tool** | Why? |
| :--- | :--- | :--- |
| Guessing | `saguaro impact --path <file>` | **Deterministic**. Lists every file, test, and build target that depends on the target file. |
| `vulture` | `saguaro deadcode` | **Confidence**. Uses reachability analysis to find true dead code. |

### Protocol
1.  **Check Impact**: Before editing a "Core" file (e.g., utility, base class), run `impact`.
2.  **Targeted Testing**: Use the impact report to decide which tests to run, rather than running the full suite (saving time).

---

## 7. Knowledge Management (Shared Brain)
**Goal**: Persist invariants and rules so other agents don't make the same mistakes.

| Generic Tool | **Saguaro Native Tool** | Why? |
| :--- | :--- | :--- |
| `notes.md` / Memory | `saguaro knowledge add` | **Structured**. Facts can be queried by other agents and validated by Sentinel. |

### Protocol
1.  **Record Decisions**: If you decide on a pattern (e.g., "Always use UTC"), add it: `saguaro knowledge add --category invariant --key "time_zone" --value "UTC"`.

---

## 8. Governance (Pre-Commit)
**Goal**: Ensure changes don't break the build or policy.

**Mandatory Step**:
Before confirming a task is done, run:
```bash
./venv/bin/saguaro verify . --engines native,ruff,semantic
```

## Example Workflow

**User**: "Fix the auth bug in login."

**Agent**:
1.  `saguaro query "login authentication logic"` -> Found `src/auth.py`
2.  `saguaro agent skeleton src/auth.py` -> See `class AuthHandler`, `def login`
3.  `saguaro agent slice AuthHandler.login` -> Read code + imports.
4.  *Agent identifies bug...*
5.  `edit_file src/auth.py` (or `saguaro agent patch`)
6.  `saguaro verify .` -> Check for regressions.
