"""Sandboxed Python code execution tool.

Safety:
    - Code runs in a separate subprocess (not exec() in the main process)
    - Hard timeout (10 seconds default) prevents infinite loops
    - No network access (optional, via environment variables)
    - Captures stdout and stderr separately

This is the most important tool for a coding agent — it lets the model
write code, run it, see the output, and iterate.
"""

import subprocess
import tempfile
from pathlib import Path

from phase7_agent.tools.base import Tool, ToolResult


class CodeExecutor(Tool):
    """Execute Python code in a sandboxed subprocess."""

    @property
    def name(self):
        return "execute_python"

    @property
    def description(self):
        return "Execute Python code and return stdout/stderr. Use for running code, testing functions, and computing results."

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            },
            "required": ["code"],
        }

    def __init__(self, timeout=10, python_path="python3"):
        self.timeout = timeout
        self.python_path = python_path

    def execute(self, code: str, **kwargs) -> ToolResult:
        """Run Python code in a subprocess.

        Args:
            code: Python source code to execute

        Returns:
            ToolResult with stdout/stderr
        """
        if not code.strip():
            return ToolResult(output="", success=True)

        # Write code to a temp file (more reliable than -c for multiline code)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [self.python_path, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={"PATH": "/usr/bin:/usr/local/bin", "HOME": "/tmp"},
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}" if output else result.stderr

            return ToolResult(
                output=output.strip() if output else "(no output)",
                success=result.returncode == 0,
                error=result.stderr.strip() if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                output="",
                success=False,
                error=f"Execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ToolResult(output="", success=False, error=str(e))
        finally:
            Path(temp_path).unlink(missing_ok=True)
