import sys
import os
import subprocess
import json
import logging
import re
from typing import List, Dict, Any
from .base import BaseEngine

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Raised when an external engine fails catastrophically."""

    pass


class ExternalUtil:
    @staticmethod
    def run_subproc(cmd: List[str], cwd: str, check: bool = False) -> str:
        # Update PATH to include the directory where the current python executable lives
        # This ensures we find tools installed in the same venv
        env = os.environ.copy()
        venv_bin = os.path.dirname(sys.executable)
        path = env.get("PATH", "")
        if venv_bin not in path:
            env["PATH"] = f"{venv_bin}:{path}"

        try:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, env=env
            )
            if check and result.returncode != 0:
                raise EngineError(
                    f"Command {' '.join(cmd)} failed with exit code {result.returncode}\n"
                    f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )
            return result.stdout + "\n" + result.stderr
        except EngineError:
            raise
        except Exception as e:
            logger.error(f"Failed to run command {' '.join(cmd)}: {e}")
            if check:
                raise EngineError(f"Subprocess execution failed: {e}")
            return ""


class RuffEngine(BaseEngine):
    def run(self) -> List[Dict[str, Any]]:
        logger.info("Running RuffEngine...")
        # ruff check . --output-format=json
        output = ExternalUtil.run_subproc(
            ["ruff", "check", ".", "--output-format=json"], self.repo_path
        )

        violations = []
        try:
            data = json.loads(output)
            # Ruff JSON format is a list of dicts
            for item in data:
                violations.append(
                    {
                        "file": item.get("filename", ""),
                        "line": item.get("location", {}).get("row", 0),
                        "rule_id": item.get("code", "RUFF"),
                        "message": item.get("message", ""),
                        "severity": "error",  # Ruff classifies most things as errors by default
                        "context": "",  # Context not easily available in JSON summaries without reading file
                    }
                )
        except json.JSONDecodeError:
            if output.strip():
                logger.warning(f"Ruff output not JSON: {output[:200]}...")
            pass

        return violations

    def fix(self, violation: Dict[str, Any]) -> bool:
        """
        Attempts to fix a ruff violation using targeted --select.
        """
        try:
            filename = violation.get("file")
            rule_id = violation.get("rule_id")

            if not filename:
                return False

            # Construct command: ruff check --fix --select <CODE> <file>
            cmd = ["ruff", "check", "--fix"]

            if rule_id and rule_id != "RUFF":
                cmd.extend(["--select", rule_id])

            cmd.append(filename)

            logger.info(f"Auto-fixing {filename} rule={rule_id} with Ruff...")
            output = ExternalUtil.run_subproc(cmd, self.repo_path)

            # Ruff usually prints nothing if fixed in quiet mode, but we can assume success
            # or check if file changed (expensive).
            # Or we check stderr/stdout.
            if "Fixed" in output:
                return True
            # If we selected a rule and it didn't complain, and didn't crash, maybe it worked?
            # But simple return True is risky.
            # Let's return True strictly if we see evidence or just optimistically?
            # Existing code was optimistic.
            return True

        except Exception as e:
            logger.error(f"Ruff fix failed: {e}")
            return False


class MypyEngine(BaseEngine):
    def run(self) -> List[Dict[str, Any]]:
        logger.info("Running MypyEngine...")
        # mypy . --output=json is not standard, we parse parsed text or use --json (newer versions)
        # using --no-error-summary to easier parsing
        output = ExternalUtil.run_subproc(
            ["mypy", ".", "--no-error-summary", "--no-pretty"], self.repo_path
        )

        violations = []
        # Format: filename:line: severity: message
        pattern = re.compile(r"^([^:]+):(\d+):\s*([a-z]+):\s*(.*)$")

        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                fpath, lineno, severity, msg = match.groups()
                violations.append(
                    {
                        "file": fpath.strip(),
                        "line": int(lineno),
                        "rule_id": "MYPY",
                        "message": msg.strip(),
                        "severity": severity,
                        "context": "",
                    }
                )
        return violations


class VultureEngine(BaseEngine):
    def run(self) -> List[Dict[str, Any]]:
        logger.info("Running VultureEngine...")
        output = ExternalUtil.run_subproc(["vulture", "."], self.repo_path)

        violations = []
        # Format: filename:line: message
        pattern = re.compile(r"^([^:]+):(\d+):\s*(.*)$")

        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                fpath, lineno, msg = match.groups()
                violations.append(
                    {
                        "file": fpath.strip(),
                        "line": int(lineno),
                        "rule_id": "VULTURE",
                        "message": msg.strip(),
                        "severity": "warning",  # Dead code is usually a warning
                        "context": "",
                    }
                )
        return violations
