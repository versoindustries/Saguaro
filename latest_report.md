# State of the Repo: saguaro_proposal
**Generated:** Mon Jan 12 11:31:01 2026

## 1. Codebase Coverage
- **Total Files:** 135
- **AST Supported:** 88
- **Languages:**
  - Python: 77
  - CMake: 1
  - Markdown: 10
  - Shell: 1
  - Text: 1
  - Pip Requirements: 1
  - YAML: 1
  - C/C++ Header: 32
  - C++: 11

## 2. Dead Code & Debt
- **Candidates:** 64
- **High Confidence:** 0
### Top Candidates to Remove:
- [0.80] `optimize_bundle` in governor.py
- [0.80] `to_dict` in context.py
- [0.80] `train_step` in encoder.py
- [0.80] `get_status` in client.py
- [0.80] `load_saguaro_library` in lib_loader.py
- [0.80] `quantum_embedding` in quantum_ops.py
- [0.80] `fused_qwt_tokenizer` in quantum_ops.py
- [0.80] `fused_coconut_bfs` in quantum_ops.py
- [0.80] `fused_text_tokenize` in quantum_ops.py
- [0.80] `clear` in tracker.py

## 3. Sentinel Health (Security & Governance)
- **Total Violations:** 17
- **Severity Breakdown:**
  - Critical: 0
  - High: 0
  - Medium: 0
  - Low: 0
  - Error: 16
  - Warning: 1

## 4. Architecture & Entry Points
- **Entry Points Detected:** 9
  - main_block: 9