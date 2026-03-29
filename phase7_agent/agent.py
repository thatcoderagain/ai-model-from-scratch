"""ReAct + Reflection Agent — the core reasoning loop.

Terminology:
    ReAct (Reasoning + Acting): An agent pattern where the model alternates between
        reasoning about what to do (THINK) and taking actions (CALL tools).
        Paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
               Yao et al. (2022). https://arxiv.org/abs/2210.03629

    Reflection: After each tool result, the model explicitly evaluates whether the
        result is correct and whether to adjust its approach. This catches errors
        that basic ReAct misses (e.g., code that runs but produces wrong output).
        Paper: "Reflexion: Language Agents with Verbal Reinforcement Learning"
               Shinn et al. (2023). https://arxiv.org/abs/2303.11366

    Agent loop:
        1. USER provides a query
        2. THINK: model reasons about the task
        3. CALL: model invokes a tool (structured JSON)
        4. OBSERVE: tool result is fed back
        5. REFLECT: model evaluates the result
        6. Repeat 2-5 or output ANSWER

    This is fundamentally how Claude Code, GitHub Copilot, and other coding
    agents work — the model plans, uses tools, checks results, and iterates.
"""

from phase7_agent.tools.base import Tool, ToolResult
from phase7_agent.function_calling import parse_model_output, validate_tool_call
from phase7_agent.memory import ConversationMemory


SYSTEM_PROMPT_TEMPLATE = """You are a helpful coding assistant. You solve problems step by step.

Available tools:
{tool_descriptions}

Format your responses using these tags:
- THINK: your reasoning about what to do next
- CALL: {{"tool": "tool_name", "args": {{...}}}} to use a tool
- ANSWER: your final response to the user

After seeing tool output (OBSERVE), reflect on whether it worked correctly.
Always test your code before giving the final answer.
"""


class ReActAgent:
    """ReAct + Reflection agent for coding tasks.

    Args:
        generate_fn: function(prompt_text) -> generated_text
            This abstracts the model — can be our local model, MLX, or even an API.
        tools: list of Tool instances
        tokenizer: tokenizer for memory management
        max_steps: maximum reasoning steps before giving up
        max_context_tokens: context window size for memory management
    """

    def __init__(self, generate_fn, tools, tokenizer,
                 max_steps=10, max_context_tokens=2048):
        self.generate_fn = generate_fn
        self.tools = {tool.name: tool for tool in tools}
        self.tokenizer = tokenizer
        self.max_steps = max_steps

        # Build system prompt with tool descriptions
        tool_desc = "\n".join(tool.to_prompt() for tool in tools)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_desc)

        self.memory = ConversationMemory(
            max_tokens=max_context_tokens,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )

    def run(self, user_query, verbose=True):
        """Run the agent on a user query.

        Args:
            user_query: the user's question or request
            verbose: print the agent's reasoning steps

        Returns:
            str: the agent's final answer
        """
        self.memory.add_message("user", user_query)

        for step in range(self.max_steps):
            # Generate model response
            context = self.memory.get_context()
            response = self.generate_fn(context)

            if verbose:
                print(f"\n--- Step {step + 1} ---")

            # Parse the response
            actions = parse_model_output(response)

            for action in actions:
                if action.type == "think":
                    if verbose:
                        print(f"THINK: {action.content}")
                    self.memory.add_message("assistant", f"THINK: {action.content}")

                elif action.type == "call":
                    if verbose:
                        print(f"CALL: {action.tool_name}({action.tool_args})")

                    # Validate
                    error = validate_tool_call(action, self.tools)
                    if error:
                        result = ToolResult(output="", success=False, error=error)
                    else:
                        # Execute
                        tool = self.tools[action.tool_name]
                        result = tool.execute(**action.tool_args)

                    # Feed result back
                    result_text = result.output if result.success else f"Error: {result.error}"
                    if verbose:
                        print(f"OBSERVE: {result_text[:200]}{'...' if len(result_text) > 200 else ''}")

                    self.memory.add_message("assistant", f"CALL: {action.content}")
                    self.memory.add_message("tool", result_text)

                elif action.type == "answer":
                    if verbose:
                        print(f"ANSWER: {action.content}")
                    self.memory.add_message("assistant", f"ANSWER: {action.content}")
                    return action.content

                elif action.type == "error":
                    if verbose:
                        print(f"PARSE ERROR: {action.content}")
                    self.memory.add_message("system",
                        f"Error parsing your response: {action.content}. "
                        "Please use THINK:, CALL:, or ANSWER: format.")

        # Ran out of steps
        return "I was unable to complete the task within the step limit. Here's what I found so far."

    def reset(self):
        """Clear conversation history for a new task."""
        self.memory.clear()
