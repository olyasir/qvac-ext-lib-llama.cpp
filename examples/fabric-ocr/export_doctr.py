#!/usr/bin/env python3
"""
Export DocTR pretrained models to GGUF format for the fabric-ocr example.

Usage:
    python export_doctr.py --model detection  --output db_mobilenet_v3.gguf
    python export_doctr.py --model recognition --output crnn_vgg16.gguf

Requirements:
    pip install python-doctr[torch] gguf numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add scripts/ to path for fabric_export
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from fabric_export import ModelExporter, export_conv_bn, export_conv, export_linear, export_lstm


# ============================================================
# Detection: DBNet + MobileNetV3-Large
# ============================================================

def export_detection(output_path):
    """Export db_mobilenet_v3_large detection model."""
    from doctr.models import db_mobilenet_v3_large

    print("Loading db_mobilenet_v3_large pretrained model...")
    model = db_mobilenet_v3_large(pretrained=True)
    model.eval()
    sd = model.state_dict()

    exporter = ModelExporter(output_path, "doctr")

    # Metadata
    exporter.writer.add_string("doctr.model_type", "detection")
    exporter.writer.add_string("doctr.backbone", "mobilenet_v3_large")
    exporter.writer.add_uint32("doctr.input_height", 1024)
    exporter.writer.add_uint32("doctr.input_width", 1024)

    # --- MobileNetV3-Large backbone ---
    # Stem: feat_extractor.0 = Conv2d(0.0) + BN(0.1)
    export_conv_bn(exporter, sd,
                   "feat_extractor.0.0", "feat_extractor.0.1",
                   "det.feat_extractor.0")

    # Inverted residual blocks: feat_extractor.{1..15}.block.{0..N}
    # Each block sequential: [expand Conv+BN], dw Conv+BN, [SE], project Conv+BN
    bneck_configs = [
        # (in_ch, exp_ch, out_ch, kernel, stride, use_se, use_hs)
        (16,  16,  16, 3, 1, False, False),   # 0 -> feat_extractor.1
        (16,  64,  24, 3, 2, False, False),   # 1 -> feat_extractor.2
        (24,  72,  24, 3, 1, False, False),   # 2 -> feat_extractor.3
        (24,  72,  40, 5, 2, True,  False),   # 3 -> feat_extractor.4
        (40, 120,  40, 5, 1, True,  False),   # 4 -> feat_extractor.5
        (40, 120,  40, 5, 1, True,  False),   # 5 -> feat_extractor.6
        (40, 240,  80, 3, 2, False, True),    # 6 -> feat_extractor.7
        (80, 200,  80, 3, 1, False, True),    # 7 -> feat_extractor.8
        (80, 184,  80, 3, 1, False, True),    # 8 -> feat_extractor.9
        (80, 184,  80, 3, 1, False, True),    # 9 -> feat_extractor.10
        (80, 480, 112, 3, 1, True,  True),    # 10 -> feat_extractor.11
        (112, 672, 112, 3, 1, True,  True),   # 11 -> feat_extractor.12
        (112, 672, 160, 5, 2, True,  True),   # 12 -> feat_extractor.13
        (160, 960, 160, 5, 1, True,  True),   # 13 -> feat_extractor.14
        (160, 960, 160, 5, 1, True,  True),   # 14 -> feat_extractor.15
    ]

    for i, (in_ch, exp_ch, out_ch, k, s, use_se, use_hs) in enumerate(bneck_configs):
        src = f"feat_extractor.{i+1}.block"
        dst = f"det.feat_extractor.1.{i}"
        out_idx = 0

        # Expand phase (only if exp_ch != in_ch)
        if exp_ch != in_ch:
            export_conv_bn(exporter, sd,
                           f"{src}.0.0", f"{src}.0.1",
                           f"{dst}.{out_idx}")
            out_idx += 1
            dw_src_idx = 1
        else:
            dw_src_idx = 0

        # Depthwise phase
        export_conv_bn(exporter, sd,
                       f"{src}.{dw_src_idx}.0", f"{src}.{dw_src_idx}.1",
                       f"{dst}.{out_idx}",
                       is_depthwise=True)
        out_idx += 1

        # SE phase: fc1/fc2 are Conv2d 1x1 in PyTorch, squeeze to 2D for GenericLinear
        if use_se:
            se_src_idx = dw_src_idx + 1
            se_prefix = f"{src}.{se_src_idx}"
            for fc_name, fc_out in [("fc1", "0"), ("fc2", "1")]:
                w = sd[f"{se_prefix}.{fc_name}.weight"].numpy()  # [OC, IC, 1, 1]
                w = w.squeeze(-1).squeeze(-1)  # -> [OC, IC]
                exporter.add_tensor(f"{dst}.{out_idx}.{fc_out}.weight", w.astype(np.float32))
                b_key = f"{se_prefix}.{fc_name}.bias"
                if b_key in sd:
                    exporter.add_tensor(f"{dst}.{out_idx}.{fc_out}.bias",
                                        sd[b_key].numpy().astype(np.float32))
            out_idx += 1
            proj_src_idx = se_src_idx + 1
        else:
            proj_src_idx = dw_src_idx + 1

        # Project phase
        export_conv_bn(exporter, sd,
                       f"{src}.{proj_src_idx}.0", f"{src}.{proj_src_idx}.1",
                       f"{dst}.{out_idx}")

    # Final conv: feat_extractor.16 = Conv(0) + BN(1), 160->960, 1x1
    export_conv_bn(exporter, sd,
                   "feat_extractor.16.0", "feat_extractor.16.1",
                   "det.feat_extractor.2")

    # --- FPN ---
    for level_idx, level_name in enumerate(["lateral2", "lateral3", "lateral4", "lateral5"]):
        export_conv_bn(exporter, sd,
                       f"fpn.in_branches.{level_idx}.0",
                       f"fpn.in_branches.{level_idx}.1",
                       f"det.fpn.{level_name}")

    for level_idx, level_name in enumerate(["output2", "output3", "output4", "output5"]):
        export_conv_bn(exporter, sd,
                       f"fpn.out_branches.{level_idx}.0",
                       f"fpn.out_branches.{level_idx}.1",
                       f"det.fpn.{level_name}")

    # --- Probability head ---
    export_conv_bn(exporter, sd,
                   "prob_head.0", "prob_head.1",
                   "det.probability_head.0")
    export_conv_bn(exporter, sd,
                   "prob_head.3", "prob_head.4",
                   "det.probability_head.1",
                   is_deconv=True)
    export_conv(exporter, sd,
                "prob_head.6",
                "det.probability_head.2",
                is_deconv=True)

    exporter.write()
    print(f"Detection model exported to {output_path}")


# ============================================================
# Recognition: CRNN + VGG16-BN
# ============================================================

def export_recognition(output_path):
    """Export crnn_vgg16_bn recognition model."""
    from doctr.models import crnn_vgg16_bn

    print("Loading crnn_vgg16_bn pretrained model...")
    model = crnn_vgg16_bn(pretrained=True)
    model.eval()
    sd = model.state_dict()

    exporter = ModelExporter(output_path, "doctr")

    # Get vocab from model
    vocab = model.cfg["vocab"]
    print(f"Vocab ({len(vocab)} chars): {vocab[:50]}...")

    # Metadata
    exporter.writer.add_string("doctr.model_type", "recognition")
    exporter.writer.add_string("doctr.backbone", "vgg16_bn")
    exporter.writer.add_string("doctr.vocab", vocab)
    exporter.writer.add_uint32("doctr.vocab_size", len(vocab))
    exporter.writer.add_uint32("doctr.seq_len", 32)
    exporter.writer.add_uint32("doctr.hidden_size", 128)
    exporter.writer.add_uint32("doctr.input_height", 32)
    exporter.writer.add_uint32("doctr.input_width", 128)

    # --- VGG16-BN backbone ---
    feat_prefix = "feat_extractor"

    # Find all Conv+BN pairs by scanning for 4D weight tensors followed by 1D BN weights
    conv_indices = []
    seen_idx = set()
    for key in sd.keys():
        if not key.startswith(feat_prefix + ".") or ".weight" not in key:
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if idx in seen_idx:
            continue
        seen_idx.add(idx)

        conv_key = f"{feat_prefix}.{idx}.weight"
        bn_key = f"{feat_prefix}.{idx+1}.weight"
        if conv_key in sd and sd[conv_key].ndim == 4 and bn_key in sd and sd[bn_key].ndim == 1:
            conv_indices.append((idx, idx + 1))

    conv_indices.sort(key=lambda x: x[0])
    print(f"Found {len(conv_indices)} Conv+BN pairs in VGG backbone")

    for out_idx, (conv_idx, bn_idx) in enumerate(conv_indices):
        export_conv_bn(exporter, sd,
                       f"{feat_prefix}.{conv_idx}",
                       f"{feat_prefix}.{bn_idx}",
                       f"rec.feat_extractor.{out_idx}")

    # --- LSTM ---
    export_lstm(exporter, sd, "decoder", "rec.encoder", num_layers=2, bidirectional=True)

    # --- Linear head ---
    export_linear(exporter, sd, "linear", "rec.decoder")

    exporter.write()
    print(f"Recognition model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export DocTR models to GGUF")
    parser.add_argument("--model", required=True, choices=["detection", "recognition"],
                        help="Which model to export")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    args = parser.parse_args()

    if args.model == "detection":
        export_detection(args.output)
    else:
        export_recognition(args.output)


if __name__ == "__main__":
    main()
