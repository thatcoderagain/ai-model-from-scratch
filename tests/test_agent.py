"""Tests for Phase 7: Agentic Coding Assistant.

Covers:
- Tool execution (code, file, shell)
- Function call parsing
- Agent loop behavior
- Memory management
"""

import tempfile
from pathlib import Path
import pytest

from phase7_agent.tools.base import ToolResult
from phase7_agent.tools.code_executor import CodeExecutor
from phase7_agent.tools.file_ops import FileReader, FileWriter
from phase7_agent.tools.shell import ShellExecutor
from phase7_agent.function_calling import parse_model_output, validate_tool_call, ParsedAction
from phase7_agent.memory import ConversationMemory
from phase7_agent.agent import ReActAgent


# ============================================================
# Tools
# ============================================================

class TestCodeExecutor:
    def test_simple_print(self):
        executor = CodeExecutor()
        result = executor.execute(code="print('hello')")
        assert result.success
        assert "hello" in result.output

    def test_math(self):
        executor = CodeExecutor()
        result = executor.execute(code="print(2 + 2)")
        assert result.success
        assert "4" in result.output

    def test_syntax_error(self):
        executor = CodeExecutor()
        result = executor.execute(code="def f(")
        assert not result.success
        assert result.error is not None

    def test_timeout(self):
        executor = CodeExecutor(timeout=2)
        result = executor.execute(code="import time; time.sleep(10)")
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_multiline_code(self):
        executor = CodeExecutor()
        code = "def add(a, b):\n    return a + b\nprint(add(3, 4))"
        result = executor.execute(code=code)
        assert result.success
        assert "7" in result.output

    def test_empty_code(self):
        executor = CodeExecutor()
        result = executor.execute(code="")
        assert result.success


class TestFileOps:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = FileWriter(allowed_dirs=[tmpdir])
            reader = FileReader(allowed_dirs=[tmpdir])

            path = str(Path(tmpdir) / "test.txt")
            write_result = writer.execute(path=path, content="hello world")
            assert write_result.success

            read_result = reader.execute(path=path)
            assert read_result.success
            assert "hello world" in read_result.output

    def test_read_nonexistent(self):
        reader = FileReader()
        result = reader.execute(path="/nonexistent/file.txt")
        assert not result.success

    def test_write_access_denied(self):
        writer = FileWriter(allowed_dirs=["/tmp/safe"])
        result = writer.execute(path="/etc/passwd", content="hack")
        assert not result.success
        assert "denied" in result.error.lower()


class TestShellExecutor:
    def test_echo(self):
        shell = ShellExecutor()
        result = shell.execute(command="echo hello")
        assert result.success
        assert "hello" in result.output

    def test_blocked_command(self):
        shell = ShellExecutor()
        result = shell.execute(command="rm -rf /")
        assert not result.success
        assert "not in allowlist" in result.error

    def test_allowed_command(self):
        shell = ShellExecutor()
        result = shell.execute(command="pwd")
        assert result.success


# ============================================================
# Function Calling Parser
# ============================================================

class TestFunctionCalling:
    def test_parse_think(self):
        actions = parse_model_output("THINK: I need to write some code")
        assert len(actions) == 1
        assert actions[0].type == "think"
        assert "write some code" in actions[0].content

    def test_parse_tool_call(self):
        text = 'CALL: {"tool": "execute_python", "args": {"code": "print(1)"}}'
        actions = parse_model_output(text)
        assert len(actions) == 1
        assert actions[0].type == "call"
        assert actions[0].tool_name == "execute_python"
        assert actions[0].tool_args["code"] == "print(1)"

    def test_parse_answer(self):
        actions = parse_model_output("ANSWER: The result is 42")
        assert len(actions) == 1
        assert actions[0].type == "answer"
        assert "42" in actions[0].content

    def test_parse_multi_step(self):
        text = """THINK: Let me calculate this
CALL: {"tool": "execute_python", "args": {"code": "print(6*7)"}}"""
        actions = parse_model_output(text)
        assert len(actions) == 2
        assert actions[0].type == "think"
        assert actions[1].type == "call"

    def test_parse_invalid_json(self):
        text = "CALL: not valid json at all"
        actions = parse_model_output(text)
        assert actions[0].type == "error"

    def test_parse_plain_text_as_answer(self):
        """Unformatted text should be treated as an answer."""
        actions = parse_model_output("Just some plain text response")
        assert len(actions) == 1
        assert actions[0].type == "answer"

    def test_validate_known_tool(self):
        executor = CodeExecutor()
        action = ParsedAction(type="call", content="", tool_name="execute_python",
                              tool_args={"code": "print(1)"})
        error = validate_tool_call(action, {"execute_python": executor})
        assert error is None

    def test_validate_unknown_tool(self):
        action = ParsedAction(type="call", content="", tool_name="unknown_tool",
                              tool_args={})
        error = validate_tool_call(action, {"execute_python": CodeExecutor()})
        assert error is not None
        assert "Unknown tool" in error

    def test_validate_missing_param(self):
        action = ParsedAction(type="call", content="", tool_name="execute_python",
                              tool_args={})  # missing "code"
        error = validate_tool_call(action, {"execute_python": CodeExecutor()})
        assert error is not None
        assert "Missing" in error


# ============================================================
# Memory
# ============================================================

class TestMemory:
    def test_add_and_retrieve(self):
        class FakeTokenizer:
            def encode(self, text):
                return text.split()

        mem = ConversationMemory(max_tokens=1000, tokenizer=FakeTokenizer(), system_prompt="You are helpful.")
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi there!")

        context = mem.get_context()
        assert "You are helpful" in context
        assert "Hello" in context
        assert "Hi there" in context

    def test_trim_old_messages(self):
        class FakeTokenizer:
            def encode(self, text):
                return text.split()

        mem = ConversationMemory(max_tokens=20, tokenizer=FakeTokenizer(), system_prompt="sys")
        for i in range(50):
            mem.add_message("user", f"message number {i}")

        # Should have trimmed old messages
        assert len(mem.messages) < 50

    def test_clear(self):
        class FakeTokenizer:
            def encode(self, text):
                return text.split()

        mem = ConversationMemory(max_tokens=1000, tokenizer=FakeTokenizer())
        mem.add_message("user", "Hello")
        mem.clear()
        assert len(mem.messages) == 0


# ============================================================
# Agent Loop
# ============================================================

class TestAgent:
    def test_demo_agent_loop(self):
        """Test agent with a scripted generate function."""
        step = [0]

        def mock_generate(prompt):
            step[0] += 1
            if step[0] == 1:
                return 'THINK: Let me run some code.\nCALL: {"tool": "execute_python", "args": {"code": "print(2+2)"}}'
            else:
                return "ANSWER: The result is 4."

        class FakeTokenizer:
            def encode(self, text):
                return text.split()

        tools = [CodeExecutor()]
        agent = ReActAgent(
            generate_fn=mock_generate,
            tools=tools,
            tokenizer=FakeTokenizer(),
            max_steps=5,
        )

        answer = agent.run("What is 2+2?", verbose=False)
        assert "4" in answer

    def test_agent_handles_tool_error(self):
        """Agent should handle tool errors gracefully."""
        step = [0]

        def mock_generate(prompt):
            step[0] += 1
            if step[0] == 1:
                return 'CALL: {"tool": "execute_python", "args": {"code": "1/0"}}'
            else:
                return "ANSWER: There was a division by zero error."

        class FakeTokenizer:
            def encode(self, text):
                return text.split()

        agent = ReActAgent(
            generate_fn=mock_generate,
            tools=[CodeExecutor()],
            tokenizer=FakeTokenizer(),
            max_steps=5,
        )

        answer = agent.run("Divide by zero", verbose=False)
        assert answer is not None  # should not crash

    def test_agent_max_steps(self):
        """Agent should stop after max_steps even without an answer."""
        def infinite_think(prompt):
            return "THINK: Still thinking..."

        class FakeTokenizer:
            def encode(self, text):
                return text.split()

        agent = ReActAgent(
            generate_fn=infinite_think,
            tools=[CodeExecutor()],
            tokenizer=FakeTokenizer(),
            max_steps=3,
        )

        answer = agent.run("Do something", verbose=False)
        assert "unable" in answer.lower() or "step limit" in answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
