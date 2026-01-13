# Integration Guide

## 1. Agent Integration (DNI)

SAGUARO connects to agents via the **Direct Native Interface (DNI)**. This is a JSON-RPC protocol over standard I/O strings.

### The Loop
1.  **Agent**: "I need to fix the login bug."
2.  **Tool Call**: `saguaro query "login authentication failure" --json`
3.  **SAGUARO**: Returns JSON list of relevant files (`[auth.py, login.tsx]`).
4.  **Agent**: Opens specific files.

### Antigravity / Gemini
Gemini is SAGUARO-aware. Ensure `GEMINI.md` references the SAGUARO protocol.

## 2. Model Context Protocol (MCP)

SAGUARO complies with the MCP Specification.

### Starting the Server
```bash
saguaro serve --mcp --port 3000
```

### Capabilities
*   **Resources**: Exposes codebase files as MCP resources (`saguaro://path/to/file`).
*   **Prompts**: Exposes templates for querying code.
*   **Tools**: Exposes `query` and `verify` tools to the MCP client.

## 3. CI/CD Integration

Use the Sentinel to gate deployments.

### GitHub Actions (`.github/workflows/sentinel.yml`)

```yaml
name: SAGUARO Sentinel
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install SAGUARO
        run: pip install saguaro-core
      - name: Run Sentinel
        run: saguaro verify
```

If `saguaro verify` returns exit code `1` (Violation), the build fails.
