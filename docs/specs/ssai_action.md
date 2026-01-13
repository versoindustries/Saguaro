# SSAI Action Specs: Patch & Impact

This document defines the schema for the "Action" layer of the SAGUARO Standard Agent Interface (SSAI).
These structures standardize how agents modify code and receive feedback on the consequences of those modifications.

## 1. The Semantic Patch (Action)

A **Semantic Patch** is an intent-based instruction to modify code. Unlike a raw unified diff, it encapsulates the *goal* and the *location* of the change, allowing the backend to apply it intelligently (and potentially auto-heal merge conflicts).

### Schema (JSON)

```json
{
  "type": "patch",
  "target_file": "src/core/engine.py",
  "operations": [
    {
      "op": "replace",
      "symbol": "Engine.process",
      "content": "    def process(self, input: str) -> dict:\n        # New implementation\n        return {\"result\": input}"
    },
    {
      "op": "insert",
      "after_symbol": "Engine.process",
      "content": "    def validate(self, data: dict) -> bool:\n        return True"
    }
  ],
  "rationale": "Updating process logic to match new API spec."
}
```

### Supported Operations
- `replace`: Swap an existing symbol (function/class) with new code.
- `insert`: Add new code relative to an existing symbol.
- `delete`: Remove a symbol.
- `update_docstring`: Modify only the docstring of a symbol.

---

## 2. The Impact Report (Feedback)

An **Impact Report** is a generated analysis of a proposed Semantic Patch *before* it is effectively committed to the disk. It predicts the consequences of the change.

### Schema (JSON)

```json
{
  "type": "impact_report",
  "patch_id": "sandbox_12345",
  "risk_score": 0.8,
  "breaking_changes": [
    {
      "severity": "high",
      "description": "Signature change in Engine.process breaks 3 callsites.",
      "locations": [
        "src/api/routes.py:45",
        "src/tests/test_engine.py:12"
      ]
    }
  ],
  "volatility_prediction": {
    "churn_probability": "high",
    "reason": "This file has changed 10 times in the last week."
  },
  "verification_status": {
    "syntax": "pass",
    "linter": "fail",
    "tests": "pending"
  }
}
```

### Goals
- **Safety**: Prevent breaking changes from being committed.
- **Feedback Loop**: Allow agents to self-correct based on specific error locations (e.g., "Signature change breaks X").
