"""Conversation memory management for the agent.

Terminology:
    Context window: The maximum number of tokens the model can see at once.
        For SmolLM2: 2048 tokens. For our scratch model: 256 tokens.
        Everything — system prompt, conversation history, tool results — must fit.

    Sliding window: When context fills up, drop the OLDEST messages to make room.
        Always keep the system prompt (it defines the agent's behavior).

    Memory management: The art of fitting as much useful context as possible
        into the fixed context window. This is why even models with 200K context
        (like Claude) still benefit from good memory management — less noise = better output.
"""


class ConversationMemory:
    """Manage conversation context within a token budget.

    Keeps system prompt + recent messages, dropping oldest when full.

    Args:
        max_tokens: maximum tokens in the context window
        tokenizer: tokenizer for counting tokens
        system_prompt: always-present system instructions
    """

    def __init__(self, max_tokens, tokenizer, system_prompt=""):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.messages = []  # list of (role, content) tuples

        # Count system prompt tokens
        self._system_tokens = len(tokenizer.encode(system_prompt)) if system_prompt else 0

    def add_message(self, role, content):
        """Add a message to the conversation history.

        Args:
            role: "user", "assistant", "tool", or "system"
            content: message text
        """
        self.messages.append((role, content))
        self._trim()

    def _trim(self):
        """Remove oldest messages if context exceeds budget."""
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)  # drop oldest

    def _total_tokens(self):
        """Estimate total token count."""
        msg_tokens = sum(len(self.tokenizer.encode(content)) for _, content in self.messages)
        return self._system_tokens + msg_tokens

    def get_context(self):
        """Build the full context string for the model.

        Returns:
            str: system prompt + conversation formatted for the model
        """
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)

        for role, content in self.messages:
            if role == "user":
                parts.append(f"USER: {content}")
            elif role == "assistant":
                parts.append(f"ASSISTANT: {content}")
            elif role == "tool":
                parts.append(f"OBSERVE: {content}")
            elif role == "system":
                parts.append(f"SYSTEM: {content}")

        return "\n\n".join(parts)

    def clear(self):
        """Clear all messages (keep system prompt)."""
        self.messages = []

    @property
    def token_count(self):
        return self._total_tokens()

    @property
    def available_tokens(self):
        return self.max_tokens - self._total_tokens()
