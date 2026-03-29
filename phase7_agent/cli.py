"""Terminal interface for the agentic coding assistant.

Usage:
    # With a fine-tuned SmolLM2 model (MLX)
    python -m phase7_agent.cli --model checkpoints/mlx_model_q4

    # With a HuggingFace model (for testing before fine-tuning)
    python -m phase7_agent.cli --hf-model HuggingFaceTB/SmolLM2-360M

    # Demo mode (uses a simple echo function instead of a real model)
    python -m phase7_agent.cli --demo

Commands:
    :reset    — clear conversation history
    :tools    — list available tools
    :quit     — exit
"""

import argparse

from phase7_agent.agent import ReActAgent
from phase7_agent.tools.code_executor import CodeExecutor
from phase7_agent.tools.file_ops import FileReader, FileWriter
from phase7_agent.tools.shell import ShellExecutor


def create_demo_generate_fn():
    """Create a simple demo generate function for testing the agent loop.

    This doesn't use a real model — it just demonstrates the agent framework.
    Replace with a real model for actual use.
    """
    step = [0]

    def generate(prompt):
        step[0] += 1
        if step[0] == 1:
            return 'THINK: Let me write and test a solution.\nCALL: {"tool": "execute_python", "args": {"code": "def hello():\\n    return \'Hello, World!\'\\nprint(hello())"}}'
        elif step[0] == 2:
            return "THINK: The code executed successfully and returned 'Hello, World!'.\nANSWER: Here's a hello world function:\n```python\ndef hello():\n    return 'Hello, World!'\n```"
        else:
            return "ANSWER: I've completed the task."

    return generate


class SimpleTokenizer:
    """Minimal tokenizer for memory management (word-level approximation)."""
    def encode(self, text):
        return text.split()  # rough approximation: 1 word ≈ 1 token


def create_hf_generate_fn(model_name, device="auto"):
    """Create a generate function using a HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                     max_new_tokens=256, temperature=0.7, top_p=0.9,
                     do_sample=True)

    def generate(prompt):
        result = pipe(prompt, return_full_text=False)
        return result[0]["generated_text"]

    return generate, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Agentic coding assistant")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no real model)")
    parser.add_argument("--hf-model", type=str, default=None, help="HuggingFace model ID")
    parser.add_argument("--model", type=str, default=None, help="Path to local MLX model")
    args = parser.parse_args()

    # --- Set up tools ---
    tools = [
        CodeExecutor(timeout=10),
        FileReader(),
        FileWriter(),
        ShellExecutor(timeout=10),
    ]

    # --- Set up generation function ---
    if args.demo:
        print("Running in DEMO mode (no real model)")
        generate_fn = create_demo_generate_fn()
        tokenizer = SimpleTokenizer()
    elif args.hf_model:
        generate_fn, tokenizer = create_hf_generate_fn(args.hf_model)
    elif args.model:
        # MLX inference (to be implemented with fine-tuned model)
        print(f"MLX model loading from {args.model} (not yet implemented)")
        print("Use --demo or --hf-model for now.")
        return
    else:
        print("No model specified. Use --demo, --hf-model, or --model.")
        print("Example: python -m phase7_agent.cli --demo")
        return

    # --- Create agent ---
    agent = ReActAgent(
        generate_fn=generate_fn,
        tools=tools,
        tokenizer=tokenizer,
        max_steps=10,
        max_context_tokens=2048,
    )

    # --- REPL ---
    print("\nLocal Coding Assistant")
    print("=" * 40)
    print("Commands: :reset, :tools, :quit")
    print("=" * 40)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query == ":quit":
            print("Bye!")
            break
        elif query == ":reset":
            agent.reset()
            if args.demo:
                agent.generate_fn = create_demo_generate_fn()
            print("Conversation cleared.")
            continue
        elif query == ":tools":
            for tool in tools:
                print(f"  {tool.to_prompt()}")
            continue

        # Run the agent
        answer = agent.run(query, verbose=True)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
