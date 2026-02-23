#!/usr/bin/env python3
"""
Fabric SDK — Shared GGUF export utilities.

Common helpers for exporting PyTorch models to GGUF format:
  - BatchNorm folding (Conv2d and ConvTranspose2d)
  - LSTM bias merging (bias_ih + bias_hh)
  - ModelExporter with auto F16/F32 type selection
  - Convenience functions: export_conv_bn, export_conv, export_linear, export_lstm

Usage:
    from fabric_export import ModelExporter, fold_bn, export_conv_bn, export_linear
"""

import numpy as np


def fold_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5, is_deconv=False):
    """Fold BatchNorm parameters into Conv weight and bias.

    For Conv2d:          weight is [OC, IC, KH, KW], BN scales dim 0 (OC).
    For ConvTranspose2d: weight is [IC, OC, KH, KW], BN scales dim 1 (OC).
    """
    scale = bn_weight / np.sqrt(bn_var + eps)

    if is_deconv:
        shape = [1] * conv_weight.ndim
        shape[1] = -1
        w = conv_weight * scale.reshape(shape)
    else:
        w = conv_weight * scale.reshape(-1, *([1] * (conv_weight.ndim - 1)))

    if conv_bias is not None:
        b = bn_bias + (conv_bias - bn_mean) * scale
    else:
        b = bn_bias - bn_mean * scale

    return w, b


def merge_lstm_bias(bias_ih, bias_hh):
    """Merge LSTM bias_ih + bias_hh into a single bias vector."""
    return bias_ih + bias_hh


def ensure_contiguous(w):
    """Ensure numpy array is contiguous."""
    return np.ascontiguousarray(w)


class ModelExporter:
    """GGUF model exporter with auto F16/F32 type selection.

    Weights (2D+) are stored as F16 for efficiency.
    Biases (1D) are stored as F32 for precision.
    """

    def __init__(self, output_path, arch_name="fabric"):
        import sys
        from pathlib import Path
        try:
            import gguf
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
            import gguf

        self.writer = gguf.GGUFWriter(output_path, arch_name)
        self.tensors = {}

    def add_tensor(self, name, data):
        """Add a tensor. Weights (2D+) as F16, biases (1D) as F32."""
        data = data.astype(np.float32)
        if data.ndim >= 2:
            data = data.astype(np.float16)
        self.tensors[name] = data

    def write(self):
        """Write all tensors and close the GGUF file."""
        for name, data in sorted(self.tensors.items()):
            self.writer.add_tensor(name, data)
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file()
        self.writer.close()
        print(f"Wrote {len(self.tensors)} tensors to GGUF")


def export_conv_bn(exporter, state_dict, conv_prefix, bn_prefix, out_name, is_deconv=False, **kwargs):
    """Export a Conv+BN pair with BN folding."""
    w = state_dict[conv_prefix + ".weight"].numpy()
    b = state_dict.get(conv_prefix + ".bias")
    b = b.numpy() if b is not None else None

    bn_w = state_dict[bn_prefix + ".weight"].numpy()
    bn_b = state_dict[bn_prefix + ".bias"].numpy()
    bn_m = state_dict[bn_prefix + ".running_mean"].numpy()
    bn_v = state_dict[bn_prefix + ".running_var"].numpy()

    w, b = fold_bn(w, b, bn_w, bn_b, bn_m, bn_v, is_deconv=is_deconv)

    exporter.add_tensor(out_name + ".weight", ensure_contiguous(w))
    exporter.add_tensor(out_name + ".bias", b)


def export_conv(exporter, state_dict, prefix, out_name, **kwargs):
    """Export a Conv layer without BN."""
    w = state_dict[prefix + ".weight"].numpy()
    exporter.add_tensor(out_name + ".weight", ensure_contiguous(w))

    b_key = prefix + ".bias"
    if b_key in state_dict:
        exporter.add_tensor(out_name + ".bias", state_dict[b_key].numpy())


def export_linear(exporter, state_dict, prefix, out_name):
    """Export a Linear layer."""
    w = state_dict[prefix + ".weight"].numpy()
    exporter.add_tensor(out_name + ".weight", w)

    b_key = prefix + ".bias"
    if b_key in state_dict:
        exporter.add_tensor(out_name + ".bias", state_dict[b_key].numpy())


def export_lstm(exporter, state_dict, lstm_prefix, out_prefix, num_layers=1, bidirectional=True):
    """Export LSTM weights with bias merging.

    Exports weight_ih, weight_hh, and merged bias for each layer/direction.
    """
    directions = ["", "_reverse"] if bidirectional else [""]

    for layer in range(num_layers):
        for direction in directions:
            suffix = f"_l{layer}{direction}"
            out_suffix = f"_l{layer}" + ("_reverse" if direction else "")

            w_ih = state_dict[f"{lstm_prefix}.weight_ih{suffix}"].numpy()
            w_hh = state_dict[f"{lstm_prefix}.weight_hh{suffix}"].numpy()
            b_ih = state_dict[f"{lstm_prefix}.bias_ih{suffix}"].numpy()
            b_hh = state_dict[f"{lstm_prefix}.bias_hh{suffix}"].numpy()

            bias = merge_lstm_bias(b_ih, b_hh)

            exporter.add_tensor(f"{out_prefix}.weight_ih{out_suffix}", w_ih)
            exporter.add_tensor(f"{out_prefix}.weight_hh{out_suffix}", w_hh)
            exporter.add_tensor(f"{out_prefix}.bias{out_suffix}", bias)
