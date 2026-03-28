"""Interactive REPL for chatting with a trained model.

Usage:
    python -m phase5_generation.interactive --checkpoint checkpoints/best.pt

    # With custom settings
    python -m phase5_generation.interactive --checkpoint checkpoints/best.pt --temperature 0.9 --top-k 50

Commands in the REPL:
    :temp 0.5     — change temperature
    :topk 40      — change top-k
    :topp 0.9     — change top-p
    :tokens 100   — change max tokens
    :settings     — show current settings
    :quit         — exit
"""

import argparse
import torch

from phase3_transformer.config import ModelConfig
from phase3_transformer.model import GPT
from phase2_tokenizer.bpe_tokenizer import BPETokenizer
from phase5_generation.generate import generate


def load_model(checkpoint_path, device=None):
    """Load a trained model from checkpoint."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    cfg = ckpt["config"]
    model_config = ModelConfig(
        vocab_size=cfg.get("vocab_size", 4096),
        n_layer=cfg.get("n_layer", 6),
        n_head=cfg.get("n_head", 6),
        n_kv_head=cfg.get("n_kv_head", 2),
        n_embd=cfg.get("n_embd", 384),
        block_size=cfg.get("block_size", 256),
        ffn_hidden=cfg.get("ffn_hidden", 1024),
        dropout=0.0,  # no dropout during inference
    )

    model = GPT(model_config)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    return model, model_config, device


def main():
    parser = argparse.ArgumentParser(description="Interactive text generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="phase2_tokenizer/vocab", help="Tokenizer path")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    # Load model and tokenizer
    print("Loading model...")
    model, config, device = load_model(args.checkpoint)
    tokenizer = BPETokenizer.load(args.tokenizer)

    stop_id = tokenizer.special_tokens.get("<|end_of_text|>")

    print(f"Model: {config.n_layer} layers, {model.count_parameters():,} params")
    print(f"Device: {device}")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print("\nType a prompt and press Enter. Commands: :temp, :topk, :topp, :tokens, :settings, :quit")
    print("-" * 60)

    # Settings
    settings = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.startswith(":"):
            parts = prompt.split()
            cmd = parts[0]
            if cmd == ":quit":
                print("Bye!")
                break
            elif cmd == ":settings":
                for k, v in settings.items():
                    print(f"  {k}: {v}")
                continue
            elif cmd == ":temp" and len(parts) > 1:
                settings["temperature"] = float(parts[1])
                print(f"  temperature = {settings['temperature']}")
                continue
            elif cmd == ":topk" and len(parts) > 1:
                settings["top_k"] = int(parts[1])
                print(f"  top_k = {settings['top_k']}")
                continue
            elif cmd == ":topp" and len(parts) > 1:
                settings["top_p"] = float(parts[1])
                print(f"  top_p = {settings['top_p']}")
                continue
            elif cmd == ":tokens" and len(parts) > 1:
                settings["max_tokens"] = int(parts[1])
                print(f"  max_tokens = {settings['max_tokens']}")
                continue
            else:
                print(f"  Unknown command: {cmd}")
                continue

        # Encode prompt
        prompt_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=device)

        # Generate
        print()
        output = generate(
            model, prompt_tensor,
            max_new_tokens=settings["max_tokens"],
            tokenizer=tokenizer,
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            top_p=settings["top_p"],
            stop_token_id=stop_id,
            stream=True,
        )


if __name__ == "__main__":
    main()
