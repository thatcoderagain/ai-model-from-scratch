"""Clean a merged model by fixing LoRA key names in the raw weight files.

Problem: When we saved the model with LoRA wrappers still in place, the weight
keys became 'q_proj.original.weight' instead of 'q_proj.weight'. This script
renames those keys and removes lora_A/lora_B so the model loads as standard Llama.

Usage:
    python -m phase6_finetune.clean_model --input checkpoints/finetune/merged_model
"""

import argparse
import json
from pathlib import Path
from safetensors.torch import load_file, save_file


def clean_model(input_dir, output_dir=None):
    """Fix LoRA key names in saved model weights."""
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir

    # Find all safetensors files
    safetensor_files = list(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        print("No .safetensors files found. Looking for .bin files...")
        import torch
        bin_files = list(input_dir.glob("*.bin"))
        if not bin_files:
            print("No model weight files found!")
            return
        # Handle .bin files
        for bf in bin_files:
            state = torch.load(bf, map_location="cpu", weights_only=True)
            state = _fix_keys(state)
            torch.save(state, output_dir / bf.name)
        print("Done (pytorch .bin files)")
        return

    # Process safetensors files
    for sf in safetensor_files:
        print(f"Processing {sf.name}...")
        state = load_file(sf)
        new_state = _fix_keys(state)

        out_path = Path(output_dir) / sf.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(new_state, out_path)

    # Fix the index file if it exists (maps key names to shard files)
    index_file = input_dir / "model.safetensors.index.json"
    if index_file.exists():
        print("Fixing weight index...")
        with open(index_file) as f:
            index = json.load(f)
        new_map = {}
        for key, shard in index.get("weight_map", {}).items():
            new_key = _fix_key_name(key)
            if new_key is not None:
                new_map[new_key] = shard
        index["weight_map"] = new_map
        out_index = Path(output_dir) / "model.safetensors.index.json"
        with open(out_index, "w") as f:
            json.dump(index, f, indent=2)

    print("Done! Model weights cleaned.")


def _fix_keys(state_dict):
    """Rename LoRA keys and remove adapter artifacts."""
    new_state = {}
    renamed = 0
    removed = 0

    for key, value in state_dict.items():
        new_key = _fix_key_name(key)
        if new_key is None:
            removed += 1
            continue
        if new_key != key:
            renamed += 1
        new_state[new_key] = value

    print(f"  Renamed: {renamed}, Removed: {removed}, Kept: {len(new_state)}")
    return new_state


def _fix_key_name(key):
    """Fix a single key name.

    'q_proj.original.weight' → 'q_proj.weight'   (renamed)
    'q_proj.lora_A'          → None                (removed)
    'q_proj.lora_B'          → None                (removed)
    'embed_tokens.weight'    → 'embed_tokens.weight' (unchanged)
    """
    # Remove LoRA adapter keys entirely
    if ".lora_A" in key or ".lora_B" in key or ".lora_dropout" in key:
        return None

    # Rename .original.weight → .weight  and  .original.bias → .bias
    if ".original." in key:
        return key.replace(".original.", ".")

    return key


def main():
    parser = argparse.ArgumentParser(description="Clean LoRA artifacts from merged model")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: overwrite input)")
    args = parser.parse_args()
    clean_model(args.input, args.output)


if __name__ == "__main__":
    main()
