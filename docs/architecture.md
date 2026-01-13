# Architecture

SAGUARO is designed as a **Quantum Codebase Operating System**. It moves beyond text search to "Holographic Resonance."

## 1. Quantum Indexing Architecture

### elastic Hyperdimensional Computing
Unlike standard vectors (fixed 1536 dims), SAGUARO uses **Elastic Dimensions**.
*   **Small Repos**: 4,096 dimensions.
*   **Enterprise**: 16,384+ dimensions.

This prevents "Semantic Collapse" where distinct concepts blur together in large codebases.

### Dark Space Buffer
SAGUARO allocates a **40% Dark Space** (zero-filled dimensions) at initialization.
*   **Purpose**: Allows the codebase to grow and evolve without requiring a full re-index.
*   **Result**: New concepts occupy the dark space, maintaining orthogonality with existing concepts.

## 2. Directory Structure

```
Project Root/
├── .saguaro/                # Local Index Store
│   ├── config.yaml          # Engine Config
│   ├── hkg/                 # Holographic Knowledge Graph
│   │   ├── vectors.npy      # Compressed HD Vectors
│   │   └── metadata.json    # Map: Hash -> File/Line
│   └── chronicle/           # Time Crystal Snapshots
│       └── snapshot_<timestamp>.pak
├── .saguaro.rules           # Sentinel Rules (User Defined)
└── .saguaro.rules.draft     # Legislator Proposals (Auto-Generated)
```

## 3. Core Components

### The Engine (`saguaro.indexing`)
*   **ContentIndexer**: Chunks files using Tree-sitter (or text split fallback).
*   **QuantumEmbedding**: Converts chunks to HD vectors using `QuantumEmbeddingOp` (C++).

### The Sentinel (`saguaro.sentinel`)
*   **Verifier**: Loads rules and scans files.
*   **RuleLoader**: Parses YAML configuration.

### The Chronicle (`saguaro.chronicle`)
*   **TimeCrystal**: Stores diffs of HD vectors over time.
*   **DriftCalculator**: Computes cosine distance between snapshots.

### The Legislator (`saguaro.legislator`)
*   **PatternMiner**: Statistical analysis of code patterns (casing, docstrings).
*   **DraftGenerator**: Outputs `.saguaro.rules.draft`.
