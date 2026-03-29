"""Structured function calling — parse and validate tool calls from model output.

Terminology:
    Function calling: The model outputs a JSON object specifying which tool to invoke
        and with what arguments. This is more reliable than parsing free-text like
        "ACT: tool_name" because JSON is unambiguous and can be validated.

    Constrained decoding: When we expect a tool call, we can force the model's output
        to start with '{' and be valid JSON. This dramatically improves reliability
        for small models that might otherwise produce malformed output.

    JSON schema validation: Check that the model's output matches the expected
        tool's parameter schema before execution. Prevents invalid arguments.

    Format:
        THINK: [reasoning about what to do]
        CALL: {"tool": "execute_python", "args": {"code": "print(2+2)"}}

        or for the final answer:
        ANSWER: [the response to the user]
"""

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedAction:
    """Parsed model output — either a tool call, a thought, or a final answer."""
    type: str          # "think", "call", "answer", "error"
    content: str       # thinking text, answer text, or error message
    tool_name: str | None = None
    tool_args: dict | None = None


def parse_model_output(text: str) -> list[ParsedAction]:
    """Parse structured output from the model.

    Expected format:
        THINK: some reasoning here
        CALL: {"tool": "tool_name", "args": {...}}
        ANSWER: final response

    Returns:
        List of parsed actions in order.
    """
    actions = []
    lines = text.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("THINK:"):
            content = line[6:].strip()
            # Collect continuation lines
            while i + 1 < len(lines) and not lines[i + 1].strip().startswith(("THINK:", "CALL:", "ANSWER:", "OBSERVE:")):
                i += 1
                content += "\n" + lines[i].strip()
            actions.append(ParsedAction(type="think", content=content))

        elif line.startswith("CALL:"):
            json_str = line[5:].strip()
            # Collect continuation lines for multi-line JSON
            while i + 1 < len(lines) and not lines[i + 1].strip().startswith(("THINK:", "CALL:", "ANSWER:", "OBSERVE:")):
                i += 1
                json_str += lines[i].strip()
            try:
                call = json.loads(json_str)
                actions.append(ParsedAction(
                    type="call",
                    content=json_str,
                    tool_name=call.get("tool"),
                    tool_args=call.get("args", {}),
                ))
            except json.JSONDecodeError:
                # Try to extract JSON from the line
                match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if match:
                    try:
                        call = json.loads(match.group())
                        actions.append(ParsedAction(
                            type="call",
                            content=match.group(),
                            tool_name=call.get("tool"),
                            tool_args=call.get("args", {}),
                        ))
                    except json.JSONDecodeError:
                        actions.append(ParsedAction(type="error", content=f"Invalid JSON: {json_str}"))
                else:
                    actions.append(ParsedAction(type="error", content=f"Invalid JSON: {json_str}"))

        elif line.startswith("ANSWER:"):
            content = line[7:].strip()
            while i + 1 < len(lines):
                i += 1
                content += "\n" + lines[i].strip()
            actions.append(ParsedAction(type="answer", content=content))

        i += 1

    # If no structured format found, treat entire text as an answer
    if not actions:
        actions.append(ParsedAction(type="answer", content=text.strip()))

    return actions


def validate_tool_call(action: ParsedAction, available_tools: dict) -> str | None:
    """Validate a tool call against available tools.

    Returns:
        None if valid, error message string if invalid.
    """
    if action.type != "call":
        return "Not a tool call"

    if action.tool_name not in available_tools:
        return f"Unknown tool: {action.tool_name}. Available: {', '.join(available_tools.keys())}"

    tool = available_tools[action.tool_name]
    schema = tool.parameters

    # Check required parameters
    required = schema.get("required", [])
    for param in required:
        if param not in (action.tool_args or {}):
            return f"Missing required parameter: {param}"

    return None
