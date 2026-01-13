---
description: Sequentially execute items from a roadmap using SAGUARO tools for context, verification, and compliance.
---

# SAGUARO Roadmap Execution Workflow

This workflow guides the agent through executing a list of tasks defined in a markdown roadmap file, ensuring strict adherence to the SAGUARO "Quantum Codebase Operating System" protocols.

## 1. SAGUARO Initialization & Indexing
Before starting any work, ensure the semantic index is fresh.

// turbo
1. Run initialization (safe to run even if already initialized):
   ```bash
   ./venv/bin/saguaro init
   ```

2. updates the semantic index with the latest codebase state:
   ```bash
   ./venv/bin/saguaro index --path .
   ```

3. Verify index health:
   ```bash
   ./venv/bin/saguaro health
   ```

## 2. Roadmap Execution Loop

For each unchecked item (`- [ ] Task Name`) in your target roadmap file:

### A. Task Scoping (Workset) & Context Retrieval
1. Create a Workset to define your focus (optional but recommended):
   ```bash
   ./venv/bin/saguaro workset create --desc "Task Name" --files "relevant/file.py"
   ```

2. Do not guess where code lives. Use the Quantum Index to locate relevant logic.

1. Formulate a query based on the task description.
2. Run the query:
   ```bash
   ./venv/bin/saguaro query "search terms for the task" --k 5
   ```
3. Read the file paths and entities returned.

### B. Plan & Analyze (Refactor/Impact)
1. If modifying an existing symbol or file, run impact analysis first:
   ```bash
   ./venv/bin/saguaro impact --path "path/to/file.py"
   # OR for specific symbol refactors
   ./venv/bin/saguaro refactor plan --symbol "MyClass"
   ```
2. Check for related entry points or build dependencies:
   ```bash
   ./venv/bin/saguaro entrypoints
   ```

### C. Implementation
1. Perform the necessary code edits, file creations, or refactors.
2. Use standard agent tools or SSAI tools for safer edits:
   ```bash
   # Get structure without reading full file
   ./venv/bin/saguaro agent skeleton path/to/file.py
   
   # Get dependency-aware context
   ./venv/bin/saguaro agent slice Symbol.name
   ```
3. Use `view_file` and `replace_file_content` for actual edits.

### D. Safety Verification & Audit (The Sentinel)
**CRITICAL**: You must verify compliance before marking a task complete.

1. Run the full verification suite:
   ```bash
   ./venv/bin/saguaro verify . --engines native,ruff,semantic
   ```
2. Run the High-Level Auditor (Governance + Impact Risk):
   ```bash
   ./venv/bin/saguaro audit
   ```
3. If violations occur, attempt auto-fix:
   ```bash
   ./venv/bin/saguaro verify --fix
   ```
4. If issues persist, fix them manually and re-verify.

### E. Chronicle & Drift Tracking
Record the "Semantic Drift" of your changes.

1. Create a snapshot/checkpoint:
   ```bash
   ./venv/bin/saguaro chronicle snapshot
   ```
2. Analyze the drift (optional, for changelog generation):
   ```bash
   ./venv/bin/saguaro chronicle diff
   ```

### F. Mark Complete
1. Edit the roadmap file to mark the item as done: `- [x] Task Name`.

## 3. Legislation (Post-Implementation)
If you implemented new patterns or significant logic, update the governance rules.

1. Draft new legislation based on recent changes:
   ```bash
   ./venv/bin/saguaro legislation --draft
   ```
