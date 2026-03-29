"""Shell command execution tool (sandboxed)."""

import subprocess
from phase7_agent.tools.base import Tool, ToolResult


# Commands that are safe to run
ALLOWED_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "find", "grep",
    "echo", "date", "whoami", "uname", "which", "env",
    "pip", "python3", "git",
}


class ShellExecutor(Tool):
    """Execute shell commands with an allowlist for safety."""

    @property
    def name(self):
        return "run_shell"

    @property
    def description(self):
        return "Run a shell command and return its output. Limited to safe commands."

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        }

    def __init__(self, timeout=10, allow_all=False):
        self.timeout = timeout
        self.allow_all = allow_all

    def execute(self, command: str, **kwargs) -> ToolResult:
        if not command.strip():
            return ToolResult(output="", success=False, error="Empty command")

        # Check allowlist
        if not self.allow_all:
            base_cmd = command.strip().split()[0]
            if base_cmd not in ALLOWED_COMMANDS:
                return ToolResult(
                    output="", success=False,
                    error=f"Command '{base_cmd}' not in allowlist. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}",
                )

        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True,
                timeout=self.timeout,
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
            return ToolResult(output="", success=False, error=f"Timed out after {self.timeout}s")
        except Exception as e:
            return ToolResult(output="", success=False, error=str(e))
