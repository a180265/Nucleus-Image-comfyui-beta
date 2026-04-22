r"""
Merge sharded safetensors into a single file.

Usage:
    python merge_shards.py --input <shard_dir> --output <output_path>

Examples:
    python merge_shards.py --input "D:\AI\Nucleus-Image\transformer" --output "output.safetensors"
    python merge_shards.py --input "D:\AI\Nucleus-Image\text_encoder" --output "output.safetensors"
"""

import argparse
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def merge_shards(shard_dir: str, output_path: str):
    """Merge sharded safetensors files into a single file."""
    # Find index file
    index_path = None
    for fname in os.listdir(shard_dir):
        if fname.endswith(".index.json"):
            index_path = os.path.join(shard_dir, fname)
            break

    if index_path is None:
        # Check if there's already a single safetensors file
        single = os.path.join(shard_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(single):
            single = os.path.join(shard_dir, "model.safetensors")
        if os.path.exists(single):
            print(f"No shards found. Copying single file: {single}")
            import shutil
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(single, output_path)
            return
        else:
            raise FileNotFoundError(f"No index.json or single safetensors found in {shard_dir}")

    print(f"Reading index: {index_path}")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    print(f"Found {len(shard_files)} shards with {len(weight_map)} tensors")

    # Estimate total size
    total_keys = len(weight_map)
    merged = {}
    loaded = 0

    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        print(f"Loading {shard_file}...", end=" ", flush=True)
        with safe_open(shard_path, framework="pt") as f:
            keys = f.keys()
            for key in keys:
                merged[key] = f.get_tensor(key)
                loaded += 1
        print(f"({loaded}/{total_keys} tensors)")

    print(f"\nTotal tensors: {len(merged)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving to {output_path}...")
    save_file(merged, output_path)
    print("Done!")

    # Verify
    out_size = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Output size: {out_size:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge safetensors shards into a single file")
    parser.add_argument("--input", required=True, help="Directory containing shard files and index.json")
    parser.add_argument("--output", required=True, help="Output safetensors file path")
    args = parser.parse_args()
    merge_shards(args.input, args.output)
