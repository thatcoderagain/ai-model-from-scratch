"""Download a pre-trained model from HuggingFace.

This is where we switch from our scratch-built 15M model to a real pre-trained model.
SmolLM2-360M already understands language and code from being trained on 11 TRILLION tokens
by HuggingFace — we just need to add our own skills on top via fine-tuning.

Terminology:
    Pre-trained model: A model that has already been trained on a massive corpus.
        It has general language understanding but isn't specialized for any task.
        Also called a "base model" or "foundation model".

    HuggingFace Hub: The main repository for open-source ML models and datasets.
        Models are stored as "safetensors" files (safe binary format for model weights).

    SmolLM2: HuggingFace's family of small language models (135M, 360M, 1.7B params).
        Architecture: Llama-style (RoPE, RMSNorm, SwiGLU, GQA) — same components we built!
        Paper: "SmolLM2: When Smol Goes Big" (arXiv 2502.02737)

    Safetensors: A safe, fast file format for storing model weights.
        Unlike pickle (which can execute arbitrary code), safetensors is pure data.

Usage:
    python -m phase6_finetune.download_model [--model SmolLM2-360M]
"""

import argparse
from pathlib import Path


SUPPORTED_MODELS = {
    "SmolLM2-135M": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "params": "135M",
        "vram_lora_fp16": "~2GB",
        "description": "Smallest SmolLM2. Good for testing, limited quality.",
    },
    "SmolLM2-360M": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M",
        "params": "360M",
        "vram_lora_fp16": "~4GB",
        "description": "Best bang for buck on RTX 3090. Recommended.",
    },
    "SmolLM2-1.7B": {
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B",
        "params": "1.7B",
        "vram_lora_fp16": "~8GB",
        "description": "Largest SmolLM2. Best quality, needs more VRAM.",
    },
}


def download_model(model_name="SmolLM2-360M", output_dir="data/models"):
    """Download a pre-trained model from HuggingFace.

    Args:
        model_name: key from SUPPORTED_MODELS
        output_dir: where to save the model

    Returns:
        Path to downloaded model directory
    """
    if model_name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    info = SUPPORTED_MODELS[model_name]
    hf_id = info["hf_id"]
    save_path = Path(output_dir) / model_name

    if save_path.exists() and any(save_path.glob("*.safetensors")):
        print(f"Model already downloaded at {save_path}")
        return save_path

    print(f"Downloading {model_name} ({info['params']} params) from {hf_id}...")
    print(f"VRAM needed for LoRA fp16: {info['vram_lora_fp16']}")
    print(f"Description: {info['description']}")

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=hf_id,
        local_dir=str(save_path),
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    print(f"Saved to {save_path}")
    return save_path


def load_pretrained(model_path, device="cpu"):
    """Load a pre-trained SmolLM2 model using HuggingFace transformers.

    Args:
        model_path: path to downloaded model directory
        device: target device

    Returns:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Download a pre-trained model")
    parser.add_argument("--model", type=str, default="SmolLM2-360M",
                        choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--output", type=str, default="data/models")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, info in SUPPORTED_MODELS.items():
            print(f"  {name:<20s} {info['params']:>6s} params | "
                  f"VRAM(LoRA): {info['vram_lora_fp16']:>6s} | {info['description']}")
        return

    download_model(args.model, args.output)


if __name__ == "__main__":
    main()
