"""Base tool interface for the agentic framework.

Terminology:
    Tool: A function the agent can invoke to interact with the world.
        Tools bridge the gap between language understanding and action.
        Examples: run code, read files, search the web.

    Tool use / Function calling: The pattern where a language model outputs
        a structured request (tool name + arguments) instead of plain text.
        The system executes the tool and feeds the result back to the model.
        This is how Claude, GPT-4, and Gemini interact with external systems.

    JSON schema: A standard format for describing the expected input structure.
        The model learns to output valid JSON matching this schema.
        Example: {"tool": "execute_python", "args": {"code": "print(2+2)"}}

    Sandboxing: Restricting what a tool can do for safety.
        Code execution runs in a subprocess with timeout and no network access.
        File operations are restricted to allowed directories.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of executing a tool."""
    output: str
    success: bool
    error: str | None = None


class Tool(ABC):
    """Base class for all agent tools.

    Subclasses must implement:
        name: unique identifier (e.g., "execute_python")
        description: what the tool does (shown to the model in the system prompt)
        parameters: JSON schema describing expected arguments
        execute(**kwargs): run the tool and return a ToolResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema for tool arguments."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        ...

    def to_prompt(self) -> str:
        """Format tool description for inclusion in the system prompt."""
        param_desc = ", ".join(
            f"{k}: {v.get('description', v.get('type', 'any'))}"
            for k, v in self.parameters.get("properties", {}).items()
        )
        return f"- {self.name}({param_desc}): {self.description}"
