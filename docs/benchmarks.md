# SAGUARO Performance Benchmarks

This document outlines the memory and performance characteristics of SAGUARO (Quantum Codebase Operating System).

## Methodology

Benchmarks were performed on **Ubuntu 24.04 LTS** with **Python 3.12**.
We generated synthetic Python codebases of varying sizes (Classes with methods, docstrings, and simple logic) and measured the **Maximum Resident Set Size (RSS)** during two operations:
1. **Indexing**: `saguaro index`
2. **Querying**: `saguaro query`

## Memory Usage

SAGUARO leverages advanced embedding models which incur a significant baseline memory cost (loading PyTorch, Transformers, and the model weights). The memory growth per Line of Code (LOC) is relatively efficient.

| Repository Size (LOC) | Operation | Memory Usage (MB) | Time (s) | Notes |
|-----------------------|-----------|-------------------|----------|-------|
| **1,000**             | Indexing  | ~7,150 MB        | ~5 s     | Baseline load dominates |
| **5,000**             | Indexing  | ~7,170 MB        | ~20 s    | Minimal growth |
| **5,000**             | Querying  | ~6,920 MB        | ~7 s     | Model inference cost |

### Extrapolated Requirements

Based on the observed growth rate (~3MB per 1,000 lines of code + constant overhead), we estimate the following requirements:

| Repository Size (LOC) | Est. Memory (GB) |
|-----------------------|------------------|
| **10,000**            | ~7.5 GB          |
| **100,000**           | ~8.0 GB          |
| **1,000,000**         | ~11.0 GB         |

> [!NOTE]
> The baseline memory usage (~7GB) is primarily due to the active embedding model and ML framework overhead. This allows SAGUARO to provide high-fidelity semantic search but requires a machine with at least **16GB RAM** for optimal performance, even for small repositories.

## Interfacing / Query Performance

Querying involves loading the index and running inference on the query string.
- **Latency**: Shallow queries on small codebases take < 10s (dominated by model load and startup time if running from CLI).
- **Resident Memory**: Querying holds a similar memory footprint to indexing, as it requires the same embedding engine.

## Recommendations

For usage with **1M+ LOC** repositories:
- **RAM**: 32GB recommended (16GB minimum).
- **Storage**: SSD essential for index latency.
- **Mode**: Use the Daemon/Server mode (future work) to avoid paying the startup/model-load cost on every CLI command.
