# Installation Guide

## Prerequisites

*   **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or WSL2.
*   **Python**: Version 3.8 or higher.
*   **Dependencies**: `numpy`, `pyyaml`, `tensorflow` (optional, for full quantum ops).

## Installation

SAGUARO is distributed as a standalone Python package.

### 1. Install from Source

Clone the repository and install in editable mode:

```bash
cd highnoon/saguaro_proposal
pip install -e .
```

### 2. Verify Installation

Check that the CLI is accessible:

```bash
saguaro --version
# Output: SAGUARO v0.1.0
```

## Initialization

Before using SAGUARO in a project, you must initialize it. This creates the `.saguaro` directory for local index storage.

```bash
cd /path/to/your/project
saguaro init
```

By default, this creates:
*   `.saguaro/`: Directory for index storage.
*   `.saguaro/config.yaml`: Default configuration.

To overwrite an existing configuration:

```bash
saguaro init --force
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'yaml'"
Ensure `PyYAML` is installed in the same environment as SAGUARO:
```bash
pip install PyYAML
```

### "TensorFlow binary is optimized..."
This is a standard warning when running on CPUs without specific instruction sets utilized by the pre-built binary. It does not affect functionality.
