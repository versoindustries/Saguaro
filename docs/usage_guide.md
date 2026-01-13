# Usage Guide

## 1. Indexing the Codebase

The core function of SAGUARO is creating a Holographic Knowledge Graph (HKG) of your code.

```bash
saguaro index
```

**Options:**
*   `--path <dir>`: Target directory (default: current directory).
*   `--verbose`: Show detailed indexing logs.

**What happens?**
1.  **Scanning**: SAGUARO scans files, respecting `.gitignore`.
2.  **Auto-Scaling**: It calculates dimensionality based on LoC.
3.  **Embedding**: Text is converted to hyperdimensional vectors (Quantum Embedding).
4.  **Storage**: Vectors are stored in `hkg/vectors.npy`.

## 2. Querying (The "Neuroserver")

Agents (and humans) can query the index to find code conceptually.

### Human Mode
```bash
saguaro query "How is authentication handled?"
```
*Output: A readable list of file paths and snippets.*

### Agent Mode (JSON)
Agents typically consume structured JSON output.

```bash
saguaro query "auth middleware" --json
```

**Output:**
```json
{
  "results": [
    {
      "file": "src/middleware/auth.py",
      "score": 0.92,
      "snippet": "class AuthMiddleware..."
    }
  ]
}
```

## 3. The Sentinel (Verification)

The Sentinel enforces project rules defined in `.saguaro.rules`.

```bash
saguaro verify
```

**Options:**
*   `path`: Specific file or directory to verify (default: all).

**Exit Codes:**
*   `0`: Success (No violations).
*   `1`: Failure (Violations found).

## 4. The Chronicle (Time Crystal)

Manage semantic snapshots of your codebase.

### Create Snapshot
```bash
saguaro chronicle snapshot --message "Refactored API"
```

### Diff Snapshots
Check semantic drift between versions:
```bash
saguaro chronicle diff
```

## 5. The Constellation (Global Logic)

Manage shared libraries.

### List Global Libraries
```bash
saguaro constellation list
```

### Link Library
Link a globally indexed library (e.g., `React`) to your local project:
```bash
saguaro constellation link React-18
```
