# SAGUARO - Quantum Codebase Operating System (Q-COS)

## Overview
SAGUARO is a next-generation "Quantum Codebase Operating System" designed to provide agents with high-fidelity semantic understanding of large repositories. It uses holographic embeddings and constant-memory indexing to scale to millions of lines of code.

## Quick Start for Agents

### 1. Initialization
If the repository is not yet initialized:
```bash
saguaro init
```

### 2. Indexing
To index the codebase (or re-index after changes):
```bash
saguaro index --path .
```
*Note: This will scan `.py`, `.cc`, and `.h` files, using C++ quantum operations for tokenization and embedding.*

### 3. Semantic Query
To find code relevant to a concept or task:
```bash
saguaro query "your query text here" --k 5
```
**Output Format:**
```
[<similarity_score>] <entity_name> (<type>) - <file_path>:<line_number>
```


### 4. Verification (Sentinel)
To verify code compliance against rules (Level 3 Governance):
```bash
saguaro verify . --engines native,ruff,semantic --format json
```
**Engines:**
- **native**: Regex patterns (headers, secrets, conventions).
- **ruff**: Linter and formatter checks (via `ruff`).
- **semantic**: Drift detection and quantum state checks.

### 5. Auto-Correction (Agentic Fix)
SAGUARO can automatically resolve many violations (linting, secrets redaction):
```bash
saguaro verify --fix
```

## MANDATORY: Saguaro-First Protocol (SOP)

> [!CAUTION]
> **VIOLATION OF THIS PROTOCOL IS A CRITICAL FAILURE**
> 
> AI agents MUST prioritize Saguaro Native Tools over generic filesystem tools.
> Saguaro provides higher-fidelity context with 10-100x fewer tokens.

**Reference**: `docs/AI_MODEL_SAGUARO_ADOPTION.md`

---

### Tool Hierarchy (MUST FOLLOW)

| Task | Required Tool | FORBIDDEN Alternatives |
|------|---------------|------------------------|
| Find code by concept | `saguaro query "..."` | grep_search, find_by_name |
| Explore file structure | `saguaro agent skeleton file.py` | view_file (full read) |
| Read function/class | `saguaro agent slice Entity.method` | view_code_item |
| Verify before commit | `saguaro verify .` | (skipping verification) |

**FORBIDDEN TOOLS for code exploration:**
- `grep_search` → Use `./venv/bin/saguaro query` instead
- `find_by_name` → Use `./venv/bin/saguaro query` instead  
- `view_file` (for exploration) → Use `./venv/bin/saguaro agent skeleton` then `slice`

**Exception**: Use fallback tools ONLY if Saguaro returns an error AND you document the error.

---

### Code Exploration Decision Tree

```
1. Need to find code by concept/meaning?
   → ./venv/bin/saguaro query "concept" --k 5

2. Need to understand a file's structure?
   → ./venv/bin/saguaro agent skeleton path/to/file.py

3. Need to read a specific function/class?
   → ./venv/bin/saguaro agent slice ClassName.method --depth 2

4. Need to verify changes before completion?
   → ./venv/bin/saguaro verify . --engines native,ruff,semantic

5. Saguaro returned an error?
   → Document the error, check index health, THEN use fallback tools
```

---

### 1. Discovery & Navigation (Replace `grep`, `find`)

**Do not guess file paths.** Find by meaning.

```bash
# ✅ CORRECT: Semantic query
./venv/bin/saguaro query "user authentication logic" --k 5

# ✅ CORRECT: Get actual code content with metadata (Structured Retrieval)
./venv/bin/saguaro query "user authentication logic" --json

# ❌ WRONG - DO NOT USE for code discovery
grep_search "auth" --path .
find_by_name "auth*.py" --directory .
```

### 2. Perception (Replace `view_file`)

**Do not read full files immediately.** Save tokens.

```bash
# ✅ CORRECT - Step 1: Get skeleton (signatures + docstrings only)
./venv/bin/saguaro agent skeleton src/core.py

# ✅ CORRECT - Step 2: Read specific entity with context
./venv/bin/saguaro agent slice MyClass.method --depth 2

# ❌ WRONG - Wastes 10-100x tokens
view_file saguaro/core.py  # Reading 500+ lines when you need 20
```

### 3. Safety Verification (MANDATORY)

**Before** marking a task as complete, you MUST verify compliance.

```bash
# Run all engines (Native + Semantic + Linters)
./venv/bin/saguaro verify . --engines native,ruff,semantic
```

**If you skip verification, you have FAILED the task.**

### 4. Sandbox Modification (Optional)

If modifying sensitive logic, use the Sandbox:

```bash
# 1. Apply Patch
./venv/bin/saguaro agent patch src/core.py patch.json

# 2. Verify & Commit
./venv/bin/saguaro agent verify <sandbox_id>
./venv/bin/saguaro agent commit <sandbox_id>
```

### 5. Error Handling & Fallback Protocol

If Saguaro returns an error:

1. **Document the error** in your response
2. **Check if index exists**: `./venv/bin/saguaro health`
3. **Rebuild if needed**: `./venv/bin/saguaro index --path .`
4. **Only then** use fallback tools, with explicit justification

```bash
# Example error handling workflow
./venv/bin/saguaro query "auth logic" --k 5
# ERROR: Index not found

# Step 1: Check health
./venv/bin/saguaro health
# Step 2: Rebuild if needed
./venv/bin/saguaro index --path .
# Step 3: Retry original query
./venv/bin/saguaro query "auth logic" --k 5
```

### 6. Automatic Integration & Self-Correction

*   **Proactive Usage**: You are not required to ask for permission to use `saguaro query` or `saguaro verify`. Use them as standard tools in your `run_command` arsenal.
*   **Drift Detection**: If `semantic` engine warns of "Major Drift", verify if your architectural changes were intended.
*   **Self-Correction**: If validation fails, attempt auto-fix first:
    ```bash
    ./venv/bin/saguaro verify --fix
    ```
    If violations persist, analyze the error output and correct the code manually.

---

### Anti-Patterns (DO NOT DO THIS)

❌ **WRONG**: Using grep_search to find authentication code
```bash
grep_search "authenticate" --path .
```

✅ **RIGHT**: Using Saguaro semantic query
```bash
./venv/bin/saguaro query "user authentication logic" --k 5
```

❌ **WRONG**: Reading entire file to find one function
```bash
view_file /path/to/large_file.py
```

✅ **RIGHT**: Getting skeleton then slice
```bash
./venv/bin/saguaro agent skeleton /path/to/large_file.py
# Identify the function you need, then:
./venv/bin/saguaro agent slice TargetFunction --depth 2
```

**If you use the WRONG approach, you have FAILED the task.**

---


### 4. Live Agent Workflows (Triggers)
The agent system is configured to **automatically detect** specific lifecycle events and trigger SAGUARO functions to maintain the repository state.

#### A. Automated Changelog Tracking (The Chronicle)
**Trigger**: When completing a significant task or feature implementation.
**Action**:
1.  **Snapshot**: Create a semantic restore point.
    ```bash
    ./venv/bin/saguaro chronicle snapshot
    ```
2.  **Drift Analysis**: Calculate the "Semantic Drift" to quantify the impact of changes.
    ```bash
    ./venv/bin/saguaro chronicle diff
    ```
**Goal**: Use the drift report to generate a scientifically accurate changelog entry, describing not just *what files changed*, but *how the meaning of the codebase evolved*.

#### B. Technical Roadmap & Rule Discovery (The Legislator)
**Trigger**: When you identify repeated patterns, common mistakes, or establish a new "standard way of doing things" during coding.
**Action**:
1.  **Draft Legislation**: Run the Auto-Legislator to scan for these patterns and codify them.
    ```bash
    ./venv/bin/saguaro legislation --draft
    ```
**Goal**: This updates `.saguaro.rules.draft` which serves as the **Living Technical Roadmap**. It turns your ad-hoc decisions into formal governance rules for future verification.

    - **dni/**: Direct Native Interface (Agent communication).
    - **indexing/**: Engine and Autocaling logic.
    - **ops/**: Python wrappers for C++ ops.
    - **storage/**: Vector store implementation.
- **src/**: C++ Source for Quantum Ops (`libsaguaro`).
- **include/**: C++ Headers.
- **tests/**: Unit and integration tests.
- **.saguaro/**: Index storage and configuration (Gitignored).

### 5. Workset Management (Task Scoping)
Organize your file context and tasks using Worksets.
```bash
# Create a workset for a specific task
./venv/bin/saguaro workset create --desc "Implement login flow" --files "src/auth.py,tests/test_auth.py"

# List active worksets
./venv/bin/saguaro workset list
```

### 6. System Diagnostics
Ensure the system is healthy before and after operations.
```bash
# Check Index Health
./venv/bin/saguaro health

# Generate Coverage Report
./venv/bin/saguaro coverage
```

### 7. Refactoring Intelligence
Plan complex changes before execution to understand impact and risk.
```bash
# Generate a dependency-aware refactor plan
./venv/bin/saguaro refactor plan --symbol "ClassName"
```

### 8. Impact & Dead Code Analysis
Identify downstream risks and cleanup opportunities.
```bash
# Analyze impact of changing a file
./venv/bin/saguaro impact --path saguaro/core.py

# Find unreachable code candidates
./venv/bin/saguaro deadcode
```

### 9. System Awareness
Understand the runtime and build structure.
```bash
# Detect runtime entry points (CLI, Rules, Hooks)
./venv/bin/saguaro entrypoints

# Visualize build dependency graph
./venv/bin/saguaro build-graph
```

### 10. Agent Knowledge Base & Audit
Share invariants between agents and perform meta-governance checks.
```bash
# Add a shared fact/invariant
./venv/bin/saguaro knowledge add --category invariant --key "auth_model" --value "RBAC"

# Run full governance audit (Sentinel + Zones + Impact)
./venv/bin/saguaro audit
```

### 11. Multi-Agent Orchestration (New)
Managed specialized agent roles and coordination.

```bash
# Run a specialized agent
./venv/bin/saguaro agent run planner --task "Refactor Auth"

# Manage Task Graph
./venv/bin/saguaro tasks --list

# Shared Structured Memory
./venv/bin/saguaro memory --list
./venv/bin/saguaro memory --write "auth_status" "migrated"
```

### 12. SSAI (Standard Agent Interface) Protocol
Use these model-agnostic tools to perceive, modify, and verify code safely via the Sandbox.

#### **Perception (Low-Token Views)**
```bash
# Get File Skeleton (Signatures only)
./venv/bin/saguaro agent skeleton src/core.py

# Get Context Slice (Dependency-Aware)
./venv/bin/saguaro agent slice MyClass.method --depth 2
```

#### **Synthesis (Generative Layer)**
```bash
# Generate a Semantic Patch from Task
# Powered by HighNoon (Native) or Dry-Run Simulation (Fallback)
./venv/bin/saguaro scribe "Refactor auth logic" --file src/auth.py --out patch.json
```

#### **Action (Sandboxed Modification)**
```bash
# 1. Apply a Semantic Patch (Returns SandboxID)
./venv/bin/saguaro agent patch src/core.py patch.json
# Output: a1b2c3d4

# 2. Verify the Sandbox State (Linters + Logic)
./venv/bin/saguaro agent verify a1b2c3d4

# 3. Predict Impact Risk
./venv/bin/saguaro agent impact a1b2c3d4

# 4. Commit to Disk (Triggers Micro-Indexing)
./venv/bin/saguaro agent commit a1b2c3d4
```

### 13. Simulation & Change Intelligence (New)
Predict the future state of the codebase.

```bash
# Generate Volatility Map (Churn Risk)
./venv/bin/saguaro simulate volatility

# Predict Regressions for changed files
./venv/bin/saguaro simulate regression --files "src/auth.py"
```

### 14. System Learning (New)
Test the self-optimizing learning components.

```bash
# Test Intent Routing
./venv/bin/saguaro route "fix the login bug"
# Output: INTENT: BUG_FIX
```

## Coding Standards
- **Python**: 3.12+, Type hinted, PEP 8.
- **C++**: C++17, TensorFlow Custom Ops API.
- **Testing**: `pytest` for Python, `bazel` (or CMake-based) for C++.

## Troubleshooting
- **Memory/Segfaults**: If `saguaro index` crashes, ensure `saguaro-core` is built with the correct TensorFlow ABI flags.
- **"Inhomogeneous shape"**: If `query` fails with this error, the index is corrupt. Run `rm -rf .saguaro` and re-index.
