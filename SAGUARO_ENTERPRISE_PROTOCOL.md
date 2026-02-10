# SAGUARO Enterprise Protocol: High-Fidelity Agentic Execution

This document defines the interface and operational standards for downstream AI agents (Antigravity, Claude, Ollama/Granite) interacting with the Saguaro Quantum-Codebase Operating System (Q-COS).

## 1. Deterministic Grounding
Agents MUST operate in a deterministic state-space to prevent reasoning drift.

**Parameter Profile:**
- **Temperature**: $10^{-14}$ (Bypass stochastic short-circuits)
- **Top_P**: $10^{-14}$ (Force logical certainty)
- **Seed**: 720,720 (Highly Composite Stack Stability)

Retrieve these via:
```json
{"method": "get_deterministic_config", "params": {"model": "granite4"}}
```

## 2. Structural Code Perception (Saguaro-First)
Do NOT read raw files for exploration. Use the perception layer to minimize token entropy.

### Agent Skeleton (Hierarchical Mapping)
Retrieve imports, class signatures, and method docstrings.
```bash
saguaro agent skeleton path/to/file.py
```

### Context Slice (Dependency-Aware)
Retrieve a symbol AND its direct imports to understand scope. Saguaro automatically includes the `get_import_graph` results in these slices.

## 3. Tiered Agentic Memory
Saguaro manages four distinct memory namespaces to separate facts from style.

| Tier | Purpose | Retrieval Method |
|------|---------|------------------|
| **Working** | Current task task/state | `memory --tier working` |
| **Episodic** | Historical events/logs | `memory --tier episodic` |
| **Semantic** | Factual code knowledge | `query "..."` |
| **Preference** | User-specific instructions | `memory --tier preference` |

## 4. Safety & Verification (Sentinel)
Before committing ANY change, agents MUST run verification.
```bash
saguaro verify . --engines native,semantic,ruff
```
- **Native**: Regex/Convention checks.
- **Semantic**: Logic drift detection (Quantum state matching).
- **Ruff**: Linting/Formatting.

## 5. Evolutionary Tracking (Chronicle)
Saguaro tracks "Semantic Drift" rather than line diffs.
- `chronicle snapshot`: Save the current quantum semantic state.
- `chronicle diff`: Compare drift scores to understand architectural impact.

---
**Standard Agent Interface (SSAI) v5.0.0**
*Engineered for Physics-Informed Deterministic AI.*
