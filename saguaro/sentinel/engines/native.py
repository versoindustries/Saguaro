import os
import logging
import fnmatch
import subprocess
from typing import List, Dict, Any
from .base import BaseEngine
from ..rules import RuleLoader

logger = logging.getLogger(__name__)


class NativeEngine(BaseEngine):
    """
    Runs the native regex-based rules defined in .saguaro.rules
    """

    def __init__(self, repo_path: str):
        super().__init__(repo_path)
        self.rules = RuleLoader.load(self.repo_path)

    def verify_file(self, file_path: str) -> List[Dict[str, Any]]:
        # Fallback for single file verification (still useful for incremental checks)
        rel_path = os.path.relpath(file_path, self.repo_path)
        violations = []

        # Filter rules by scope
        active_rules = [r for r in self.rules if fnmatch.fnmatch(rel_path, r.scope)]

        if not active_rules:
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            for rule in active_rules:
                matches = rule.check(content)
                for line_num, line_content in matches:
                    violations.append(
                        {
                            "file": rel_path,
                            "line": line_num,
                            "rule_id": rule.id,
                            "message": rule.message,
                            "severity": rule.severity,
                            "context": line_content,
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to verify {rel_path}: {e}")

        return violations

    def run(self) -> List[Dict[str, Any]]:
        logger.info("Running NativeEngine (Optimized)...")
        violations = []

        # Check for git
        has_git = os.path.isdir(os.path.join(self.repo_path, ".git"))

        if not has_git:
            logger.info("No .git directory found. Using system 'grep' fallback.")
            return self._run_grep_fallback()

        # Fast path: Use git grep for each rule
        for rule in self.rules:
            try:
                # Construct command: git grep -n -I -P "pattern" [pathspec]
                cmd = ["git", "grep", "-n", "-I", "-P", rule.pattern]

                # Scope handling
                # rule.scope is glob pattern. simple globs work in git grep pathspec
                if rule.scope and rule.scope != "*":
                    cmd.append(rule.scope)

                result = subprocess.run(
                    cmd, cwd=self.repo_path, capture_output=True, text=True
                )

                # git grep returns 0 on match, 1 on no match, 2 on error
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        # Output format: file:line:content
                        # content might contain colons, so split only first 2
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            violations.append(
                                {
                                    "file": parts[0],
                                    "line": int(parts[1]),
                                    "rule_id": rule.id,
                                    "message": rule.message,
                                    "severity": rule.severity,
                                    "context": parts[2].strip(),
                                }
                            )
                elif result.returncode > 1:
                    logger.debug(f"git grep error for rule {rule.id}: {result.stderr}")

            except Exception as e:
                logger.warning(f"Failed to run grep for rule {rule.id}: {e}")

        return violations

    def _run_grep_fallback(self) -> List[Dict[str, Any]]:
        """Fallback to system grep for non-git repositories."""
        violations = []

        for rule in self.rules:
            try:
                # Use grep -rInP (recursive, line-number, binary-files=without-match, Perl-regex)
                # Need to handle excludes manually or using grep flags?
                # grep doesn't have easy ignore pattern list like git grep.
                # But we can assume standard ignore?
                # Or just let it scan everything if it's performant enough.
                # Use --exclude-dir for common junk.

                cmd = ["grep", "-rInP", rule.pattern]

                # Basic excludes
                excludes = [
                    ".git",
                    ".saguaro",
                    "__pycache__",
                    "node_modules",
                    "venv",
                    "build",
                ]
                for ex in excludes:
                    cmd.extend(["--exclude-dir", ex])

                # Scope? "grep" accepts path arguments at the end.
                if rule.scope and rule.scope != "*":
                    # If scope is glob, might need shell expansion or find.
                    # For simplicity, if scope is simple path, append it.
                    # If scope has wildcards, grep might not handle it directly as path arg.
                    # Let's just scan root "." and rely on pattern, or if scope is specific dir, use it.
                    pass

                cmd.append(self.repo_path)

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            # grep path is absolute or relative depending on arg.
                            # We passed absolute self.repo_path, so grep output likely absolute?
                            # Or if we pass ., it is relative.
                            fpath = parts[0]
                            rel_path = os.path.relpath(fpath, self.repo_path)

                            violations.append(
                                {
                                    "file": rel_path,
                                    "line": int(parts[1]),
                                    "rule_id": rule.id,
                                    "message": rule.message,
                                    "severity": rule.severity,
                                    "context": parts[2].strip(),
                                }
                            )

            except Exception as e:
                logger.warning(f"Grep failed for rule {rule.id}: {e}")

        return violations

    def fix(self, violation: Dict[str, Any]) -> bool:
        """
        Attempts to fix a native rule violation using regex substitution.
        """
        import re

        rule_id = violation.get("rule_id")
        rule = next((r for r in self.rules if r.id == rule_id), None)

        if not rule or not rule.replacement:
            logger.debug(f"No replacement pattern for rule {rule_id}")
            return False

        file_path = os.path.join(self.repo_path, violation.get("file"))
        line_num = violation.get("line")  # 1-based

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if line_num > len(lines):
                return False

            # Fix only the specific line
            target_line = lines[line_num - 1]
            # Replace pattern with replacement
            new_line = re.sub(rule.pattern, rule.replacement, target_line)

            if new_line != target_line:
                lines[line_num - 1] = new_line
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                logger.info(f"Fixed {rule_id} in {violation.get('file')}:{line_num}")
                return True

        except Exception as e:
            logger.error(f"Native fix failed for {file_path}: {e}")

        return False
