"""Terminal interface for the agentic coding assistant.

Usage:
    # With MLX quantized model (fastest on MacBook)
    python -m phase7_agent.cli --model checkpoints/mlx_model_q4

    # With a HuggingFace model (merged or from hub)
    python -m phase7_agent.cli --hf-model checkpoints/finetune/merged_model
    python -m phase7_agent.cli --hf-model HuggingFaceTB/SmolLM2-360M

    # Demo mode (uses a scripted function instead of a real model)
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
    """Create a simple demo generate function for testing the agent loop."""
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
        return text.split()


def create_mlx_generate_fn(model_path):
    """Create a generate function using MLX (fast on Apple Silicon)."""
    from mlx_lm import load, generate as mlx_generate

    print(f"Loading MLX model from {model_path}...")
    model, tokenizer = load(model_path)
    print(f"Model loaded.")

    def generate(prompt):
        response = mlx_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=150,
        )
        if not response:
            return ""

        # Small models don't stop cleanly — they hallucinate extra turns.
        # Cut at the first fake "USER:" or repeated "CODE:" block.
        stop_markers = ["\nUSER:", "\nUser:", "\n\nUSER", "\n\n\n"]
        for marker in stop_markers:
            idx = response.find(marker)
            if idx != -1:
                response = response[:idx]

        return response.strip()

    return generate, tokenizer


def create_hf_generate_fn(model_name):
    """Create a generate function using a HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
    ).to(device)
    model.eval()

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
                temperature=0.7, top_p=0.9, do_sample=True,
            )
        # Decode only the generated part (not the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Agentic coding assistant")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no real model)")
    parser.add_argument("--hf-model", type=str, default=None,
                        help="HuggingFace model path or ID (e.g., checkpoints/finetune/merged_model)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to MLX model (e.g., checkpoints/mlx_model_q4)")
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
    elif args.model:
        generate_fn, tokenizer = create_mlx_generate_fn(args.model)
    elif args.hf_model:
        generate_fn, tokenizer = create_hf_generate_fn(args.hf_model)
    else:
        print("No model specified. Options:")
        print("  --model checkpoints/mlx_model_q4          (MLX, fastest on MacBook)")
        print("  --hf-model checkpoints/finetune/merged_model  (HuggingFace)")
        print("  --demo                                     (no model, test framework)")
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
