"""Convert a fine-tuned model to MLX format for fast MacBook inference.

Terminology:
    MLX: Apple's ML framework, built specifically for Apple Silicon (M1-M5).
        Provides 2-3x faster inference than PyTorch MPS on the same hardware.
        Uses unified memory efficiently — the GPU can access CPU memory directly.

    Why convert: PyTorch models run on MPS (Metal Performance Shaders) but MLX
        is optimized for Apple Silicon's unified memory architecture. After converting,
        we get faster inference for the interactive REPL and agent.

    Quantization (done separately in quantize.py): Further compress the model
        from float16 to 4-bit integers. Reduces model size ~4x and speeds up inference.

Usage:
    python -m phase6_finetune.convert_to_mlx --input checkpoints/finetune/merged_model --output checkpoints/mlx_model
"""

import argparse
from pathlib import Path


def convert_hf_to_mlx(input_dir, output_dir):
    """Convert a HuggingFace model to MLX format.

    Uses mlx-lm's convert utility which handles:
        - Weight format conversion (safetensors → MLX npz)
        - Config translation (HF config → MLX config)
        - Tokenizer copying
    """
    try:
        from mlx_lm import convert
    except ImportError:
        print("MLX not installed. Install with: uv pip install 'mlx>=0.18' 'mlx-lm>=0.20'")
        print("Note: MLX only works on Apple Silicon Macs.")
        return None

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {input_dir} → {output_dir}")
    convert(str(input_dir), mlx_path=str(output_dir))
    print(f"MLX model saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert model to MLX format")
    parser.add_argument("--input", type=str, required=True, help="Path to HuggingFace model")
    parser.add_argument("--output", type=str, default="checkpoints/mlx_model", help="Output path")
    args = parser.parse_args()

    convert_hf_to_mlx(args.input, args.output)


if __name__ == "__main__":
    main()
