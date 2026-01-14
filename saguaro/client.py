"""
SAGUARO Client Library
Python client for agents to interact with the SAGUARO DNI Server.
"""

import subprocess
import json
import logging
import os
import atexit
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SAGUAROClient:
    def __init__(self, repo_path: str = ".", saguaro_cmd: str = "saguaro"):
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_cmd = saguaro_cmd
        self.process = None
        self._seq = 0

        self.start()
        atexit.register(self.stop)

    def start(self):
        """Starts the SAGUARO DNI server subprocess."""
        if isinstance(self.saguaro_cmd, list):
            cmd = self.saguaro_cmd + ["serve"]
        else:
            cmd = [self.saguaro_cmd, "serve"]

        logger.info(f"Starting SAGUARO DNI: {' '.join(cmd)}")
        try:
            # We need to run in the venv of the repo usually?
            # Or assume 'saguaro' is in PATH.
            # For now, simplistic subprocess
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Log file handles stderr usually
                cwd=self.repo_path,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Initialize
            self._send_request("initialize", {"path": self.repo_path})

        except Exception as e:
            logger.error(f"Failed to start SAGUARO: {e}")
            raise e

    def stop(self):
        """Stops the SAGUARO server."""
        if self.process:
            self.process.terminate()
            self.process = None

    def _send_request(self, method: str, params: Dict[str, Any] = None) -> Any:
        if not self.process:
            raise RuntimeError("SAGUARO is not running")

        params = params or {}
        self._seq += 1
        req_id = self._seq

        req = {"jsonrpc": "2.0", "method": method, "params": params, "id": req_id}

        try:
            line_str = json.dumps(req) + "\n"
            self.process.stdin.write(line_str)
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise RuntimeError("SAGUARO server closed connection")

            response = json.loads(response_line)

            if "error" in response:
                raise RuntimeError(f"SAGUARO Error: {response['error']}")

            if response.get("id") != req_id:
                # In simple synch client, this shouldn't happen unless protocol violation
                logger.warning(
                    f"ID mismatch: expected {req_id}, got {response.get('id')}"
                )

            return response.get("result")

        except BrokenPipeError:
            self.process = None
            raise RuntimeError("SAGUARO server crashed")

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the codebase index.
        """
        res = self._send_request("query", {"text": text, "k": k})
        return res.get("results", [])

    def read_node(self, entity: Dict[str, Any]) -> str:
        """
        Reads source code for an entity returned by query.
        """
        res = self._send_request(
            "read_node",
            {
                "file": entity.get("file"),
                "start_line": entity.get("line"),  # Start line from index
                "end_line": entity.get(
                    "end_line"
                ),  # End line from index (if available)
            },
        )
        return res.get("content", "")

    def get_status(self):
        return self._send_request("status")
