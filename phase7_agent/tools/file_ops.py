"""File reading/writing tool for the agent."""

from pathlib import Path
from phase7_agent.tools.base import Tool, ToolResult


class FileReader(Tool):
    """Read file contents."""

    @property
    def name(self):
        return "read_file"

    @property
    def description(self):
        return "Read the contents of a file. Returns the file text or an error."

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        }

    def __init__(self, allowed_dirs=None):
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs] if allowed_dirs else None

    def execute(self, path: str, **kwargs) -> ToolResult:
        file_path = Path(path).resolve()

        if self.allowed_dirs and not any(str(file_path).startswith(str(d)) for d in self.allowed_dirs):
            return ToolResult(output="", success=False, error=f"Access denied: {path}")

        if not file_path.exists():
            return ToolResult(output="", success=False, error=f"File not found: {path}")

        try:
            content = file_path.read_text(encoding="utf-8")
            if len(content) > 10000:
                content = content[:10000] + f"\n... (truncated, {len(content)} chars total)"
            return ToolResult(output=content, success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=str(e))


class FileWriter(Tool):
    """Write content to a file."""

    @property
    def name(self):
        return "write_file"

    @property
    def description(self):
        return "Write content to a file. Creates the file if it doesn't exist."

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    def __init__(self, allowed_dirs=None):
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs] if allowed_dirs else None

    def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        file_path = Path(path).resolve()

        if self.allowed_dirs and not any(str(file_path).startswith(str(d)) for d in self.allowed_dirs):
            return ToolResult(output="", success=False, error=f"Access denied: {path}")

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ToolResult(output=f"Written {len(content)} chars to {path}", success=True)
        except Exception as e:
            return ToolResult(output="", success=False, error=str(e))
