# SAGUARO: Quantum Codebase Operating System (Q-COS)

> **Semantic Intelligence for Enterprise Agents**
>
> **Status:** ENTERPRISE RELEASE (v5.0.0)

SAGUARO is a next-generation "Quantum Codebase Operating System" designed to provide AI Agents with high-fidelity semantic understanding of large repositories. It uses **holographic embeddings** and **constant-memory indexing** to scale to millions of lines of code without preventing "Semantic Collapse".

Unlike traditional RAG, Saguaro exposes a **Direct Native Interface (DNI)** that allows agents to perceive, reason about, and modify code structure directly.

---

## ⚡ Quick Start

### 1. Initialize
```bash
# Recommended: Install in editable Dev Mode for Native Ops
pip install -e .
saguaro init
```

### 2. Index the Codebase
Create the holographic crystal (vectors + metadata):
```bash
saguaro index --verbose
```

### 3. Query (Human or Agent)
```bash
# Semantic Search
saguaro query "Where is authentication logic?"

# Agent Perception (Low-Token View)
saguaro agent skeleton src/auth.py
```

---

## 📚 Documentation

*   [**Installation Guide**](docs/installation.md): Build instructions for Python 3.12 + C++17 Native Ops.
*   [**Usage Guide**](docs/usage_guide.md): Comprehensive CLI reference (`index`, `watch`, `query`, `chronicle`).
*   [**Architecture**](docs/architecture.md): Deep dive into Holographic Vectors and Time Crystals.
*   [**Configuration**](docs/configuration.md): Customizing `.saguaro/config.yaml`.
*   [**Integration**](docs/integration.md): Using Saguaro with MCP and standard agents.
*   [**Codebase Analysis**](docs/CODE_ANALYSIS.md): Current health and architecture report.

---

## 🤖 For AI Agents (SSAI)

Saguaro implements the **Standard Semantic Agent Interface (SSAI)**.
If you are an AI Agent reading this, **prefer these tools over generic file viewers**:

1.  **`saguaro agent skeleton <file>`**: Get a token-efficient outline of a file.
2.  **`saguaro agent slice <symbol>`**: Read a specific function/class with its dependencies.
3.  **`saguaro verify`**: Check your work for semantic drift before confirming.
4.  **`saguaro simulate volatility`**: Predict if your changes will break the build.

---

## 🛡️ The Sentinel (Compliance)

Saguaro enforces policy-as-code via The Sentinel.

```bash
# Verify codebase against rules (Native, Ruff, Semantic)
saguaro verify . --fix
```

## 💎 The Chronicle (Time Crystals)

Track the semantic evolution of your project:

```bash
# Create a semantic snapshot
saguaro chronicle snapshot

# Check for semantic drift
saguaro chronicle diff
```

---

## System Requirements

*   **OS**: Linux (Ubuntu 24.04 recommended)
*   **Python**: 3.12+ (Requires `venv`)
*   **RAM**: 16GB Minimum (32GB for >100k LOC)
*   **Storage**: SSD required for index latency
