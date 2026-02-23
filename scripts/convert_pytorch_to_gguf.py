#!/usr/bin/env python3
"""
Convert a PyTorch state_dict (.pt/.pth/.bin) to GGUF format for use with Fabric SDK.

Features:
- Writes all tensors with original names
- BatchNorm fusion: (weight, bias, running_mean, running_var) -> (scale, shift)
- Conv weight transposition: PyTorch [OC,IC,KH,KW] -> ggml [KW,KH,IC,OC]
- Supports f32 and f16 output

Usage:
    python scripts/convert_pytorch_to_gguf.py model.pt output.gguf [--type f16] [--arch mnist_cnn]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian u32
GGUF_VERSION = 3

# GGUF types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING  = 8

# GGML types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def write_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_string(f, key: str, val: str):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    write_string(f, val)


def write_kv_u32(f, key: str, val: int):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", val))


def write_kv_i32(f, key: str, val: int):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_INT32))
    f.write(struct.pack("<i", val))


def write_kv_f32(f, key: str, val: float):
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<f", val))


def fuse_batchnorm(state_dict: dict, prefix: str, eps: float = 1e-5) -> dict:
    """Fuse BatchNorm into scale + shift.

    Given BN params (weight/gamma, bias/beta, running_mean, running_var),
    computes:
        scale = gamma / sqrt(running_var + eps)
        shift = beta - gamma * running_mean / sqrt(running_var + eps)
    """
    gamma = state_dict.get(f"{prefix}.weight")
    beta  = state_dict.get(f"{prefix}.bias")
    mean  = state_dict.get(f"{prefix}.running_mean")
    var   = state_dict.get(f"{prefix}.running_var")

    if mean is None or var is None:
        return {}

    if gamma is None:
        gamma = torch.ones_like(mean)
    if beta is None:
        beta = torch.zeros_like(mean)

    inv_std = 1.0 / torch.sqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - gamma * mean * inv_std

    return {
        f"{prefix}.scale": scale,
        f"{prefix}.shift": shift,
    }


def transpose_conv_weight(tensor: np.ndarray) -> np.ndarray:
    """Transpose conv weight from PyTorch layout to ggml layout.

    PyTorch conv2d: [OC, IC, KH, KW]
    ggml conv2d:    [KW, KH, IC, OC]  (column-major, so we reverse dims)
    """
    if tensor.ndim == 4:
        # [OC, IC, KH, KW] -> [KW, KH, IC, OC]
        return np.ascontiguousarray(tensor.transpose(3, 2, 1, 0))
    elif tensor.ndim == 3:
        # Conv1d: [OC, IC, KW] -> [KW, IC, OC]
        return np.ascontiguousarray(tensor.transpose(2, 1, 0))
    return tensor


def is_conv_weight(name: str, tensor: np.ndarray) -> bool:
    """Heuristic to detect conv weights."""
    return "weight" in name and tensor.ndim >= 3


def is_batchnorm(name: str) -> bool:
    """Check if this key belongs to a BatchNorm layer."""
    bn_keys = (".running_mean", ".running_var", ".num_batches_tracked")
    return any(name.endswith(k) for k in bn_keys)


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to GGUF")
    parser.add_argument("input", help="Input PyTorch file (.pt, .pth, .bin)")
    parser.add_argument("output", help="Output GGUF file")
    parser.add_argument("--type", choices=["f32", "f16"], default="f32",
                        help="Output tensor type (default: f32)")
    parser.add_argument("--arch", default="generic",
                        help="Architecture name stored in metadata")
    parser.add_argument("--fuse-bn", action="store_true",
                        help="Fuse BatchNorm layers into scale+shift")
    parser.add_argument("--transpose-conv", action="store_true",
                        help="Transpose conv weights to ggml layout")
    args = parser.parse_args()

    # Load state dict
    print(f"Loading {args.input}...")
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
    elif isinstance(data, dict) and "model_state_dict" in data:
        state_dict = data["model_state_dict"]
    elif isinstance(data, dict):
        state_dict = data
    else:
        # Assume it's a raw model — get state_dict
        state_dict = data.state_dict() if hasattr(data, "state_dict") else data

    # Fuse BatchNorm if requested
    if args.fuse_bn:
        bn_prefixes = set()
        for key in list(state_dict.keys()):
            if key.endswith(".running_mean"):
                prefix = key[:-len(".running_mean")]
                bn_prefixes.add(prefix)

        for prefix in sorted(bn_prefixes):
            fused = fuse_batchnorm(state_dict, prefix)
            if fused:
                print(f"  Fusing BatchNorm: {prefix}")
                # Remove original BN keys
                for suffix in (".weight", ".bias", ".running_mean", ".running_var", ".num_batches_tracked"):
                    state_dict.pop(f"{prefix}{suffix}", None)
                # Add fused keys
                state_dict.update(fused)

    # Remove num_batches_tracked (not needed)
    for key in list(state_dict.keys()):
        if key.endswith(".num_batches_tracked"):
            del state_dict[key]

    # Convert to numpy
    ggml_type = GGML_TYPE_F16 if args.type == "f16" else GGML_TYPE_F32
    np_dtype = np.float16 if args.type == "f16" else np.float32

    tensors = {}
    for name, param in state_dict.items():
        arr = param.detach().float().numpy()

        # Transpose conv weights if requested
        if args.transpose_conv and is_conv_weight(name, arr):
            print(f"  Transposing conv: {name} {arr.shape}", end="")
            arr = transpose_conv_weight(arr)
            print(f" -> {arr.shape}")

        arr = arr.astype(np_dtype)
        tensors[name] = arr

    # Metadata
    metadata = {
        "general.architecture": (GGUF_TYPE_STRING, args.arch),
        "general.name": (GGUF_TYPE_STRING, Path(args.input).stem),
    }

    n_kv = len(metadata)
    n_tensors = len(tensors)

    print(f"Writing {args.output} ({n_tensors} tensors, {n_kv} metadata keys)...")

    with open(args.output, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        # Metadata KV pairs
        for key, (vtype, val) in metadata.items():
            if vtype == GGUF_TYPE_STRING:
                write_kv_string(f, key, val)
            elif vtype == GGUF_TYPE_UINT32:
                write_kv_u32(f, key, val)
            elif vtype == GGUF_TYPE_INT32:
                write_kv_i32(f, key, val)
            elif vtype == GGUF_TYPE_FLOAT32:
                write_kv_f32(f, key, val)

        # Tensor info
        # For each tensor: name, n_dims, shape (ne), type, offset
        data_start_offset = 0
        tensor_offsets = []

        for name, arr in tensors.items():
            write_string(f, name)
            n_dims = len(arr.shape)
            f.write(struct.pack("<I", n_dims))
            # ggml ne[0] = innermost (fastest) dim = last numpy dim → reverse
            for dim in reversed(arr.shape):
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", ggml_type))

            # Compute offset (aligned to 32 bytes)
            tensor_offsets.append(data_start_offset)
            f.write(struct.pack("<Q", data_start_offset))

            nbytes = arr.nbytes
            data_start_offset += nbytes
            # Align to 32 bytes
            padding = (32 - (data_start_offset % 32)) % 32
            data_start_offset += padding

        # Pad to alignment before tensor data
        current_pos = f.tell()
        alignment = 32
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b"\x00" * padding)

        # Tensor data
        for i, (name, arr) in enumerate(tensors.items()):
            f.write(arr.tobytes())
            # Pad to 32 bytes
            nbytes = arr.nbytes
            padding = (32 - (nbytes % 32)) % 32
            f.write(b"\x00" * padding)

    output_size = Path(args.output).stat().st_size
    print(f"Done! Output: {output_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
