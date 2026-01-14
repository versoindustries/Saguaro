"""
Test script to simulate an Agent interacting with SAGUARO DNI via the Client Library.
"""

import sys
import os

# Ensure we can import from local
sys.path.append(os.getcwd())

from saguaro.client import SAGUAROClient


def simulate_agent_session():
    print("--- Starting Agent Session ---")

    # 1. Initialize Client (starts DNI server as subprocess)
    print("[Agent] Initializing SAGUARO Client...")
    # We must point to the 'bin/saguaro' equivalent or just run python -m saguaro.cli
    # Since we didn't install it yet, let's use the module invoc
    # We'll wrapper the "command" to be python -m ...

    python_exe = sys.executable
    # cmd_prefix = f"{python_exe} -m saguaro.cli"

    # Needs to run in unbuffered mode equivalent or use list
    # SAGUAROClient expects a command string or list?
    # subprocess.Popen(cmd, ...)
    # If we pass a list, Popen handles it.

    # Use list for command so Client appends "serve" correctly
    client = SAGUAROClient(repo_path=".", saguaro_cmd=[python_exe, "-m", "saguaro.cli"])

    # Manual handshake
    print("[Agent] Sending Initialize...")
    res = client._send_request("initialize", {"path": os.path.abspath(".")})
    print(f"[SAGUARO] Initialize Result: {res}")

    # 2. Query
    query_text = "quantum chunking"
    print(f"[Agent] Querying: '{query_text}'")
    results = client.query(query_text, k=3)

    print(f"[Agent] Got {len(results)} results:")
    for i, res in enumerate(results):
        print(f"  {i + 1}. [{res.get('score', 0):.4f}] {res['name']} ({res['file']})")

    if not results:
        print(
            "[Agent] No results found! Indexing might have failed or persisted incorrectly."
        )
        sys.exit(1)

    # 3. Read Node (Source Retrieval)
    target = results[0]
    print(f"[Agent] Requesting source for: {target['name']}")

    source = client.read_node(target)

    print("--- Source Code Preview ---")
    lines = source.splitlines()
    for line in lines[:5]:
        print(line)
    print("...")
    print(f"--- End Source ({len(lines)} lines) ---")

    print("[Agent] Session Complete.")
    client.stop()


if __name__ == "__main__":
    simulate_agent_session()
