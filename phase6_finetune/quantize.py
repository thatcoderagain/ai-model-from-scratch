"""Quantize a model to 4-bit for fast, memory-efficient inference.

Terminology:
    Quantization: Reducing the numerical precision of model weights.
        float32 (4 bytes per weight) → float16 (2 bytes) → int4 (0.5 bytes).
        A 360M param model: fp32 = 1.4GB, fp16 = 720MB, int4 = ~90MB.

    Why quantize:
        1. Smaller model → fits in less RAM
        2. Less memory bandwidth → faster inference (memory-bound on Apple Silicon)
        3. Quality loss is surprisingly small: 4-bit retains ~95% of fp16 quality
        4. Makes it practical to run on MacBook's 16GB RAM

    Post-training quantization (PTQ): Quantize AFTER training (what we do here).
        No additional training needed — just convert the weights.

    4-bit quantization (Q4): Each weight stored as a 4-bit integer + per-group
        scale factor. "Groups" of 32-128 weights share one scale, which preserves
        more precision than a single global scale.

    GGUF: A popular quantization format (used by llama.cpp). We use MLX's built-in
        quantization instead since we're deploying on Apple Silicon.

Usage:
    python -m phase6_finetune.quantize --input checkpoints/mlx_model --bits 4
"""

import argparse
from pathlib import Path


def quantize_mlx_model(input_dir, output_dir=None, bits=4, group_size=64):
    """Quantize an MLX model to lower precision.

    Args:
        input_dir: path to MLX format model
        output_dir: output path (default: input_dir + '_q4')
        bits: quantization bits (4 or 8)
        group_size: number of weights sharing one scale factor

    Returns:
        Path to quantized model
    """
    try:
        from mlx_lm import convert
    except ImportError:
        print("MLX not installed. Install with: uv pip install 'mlx>=0.18' 'mlx-lm>=0.20'")
        return None

    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_q{bits}"
    output_dir = Path(output_dir)

    # mlx_lm.convert creates the output dir itself — remove if it exists
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    print(f"Quantizing {input_dir} → {output_dir}")
    print(f"  Bits: {bits}, Group size: {group_size}")

    # mlx-lm's convert handles quantization when q_bits is specified
    convert(
        str(input_dir),
        mlx_path=str(output_dir),
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
    )

    # Report size savings
    input_size = sum(f.stat().st_size for f in input_dir.rglob("*") if f.is_file())
    output_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    ratio = input_size / max(output_size, 1)
    print(f"  Input:  {input_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {output_size / 1024 / 1024:.1f} MB")
    print(f"  Compression: {ratio:.1f}x")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Quantize model for fast inference")
    parser.add_argument("--input", type=str, required=True, help="Path to MLX model")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    args = parser.parse_args()

    quantize_mlx_model(args.input, args.output, args.bits, args.group_size)


if __name__ == "__main__":
    main()
