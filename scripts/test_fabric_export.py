#!/usr/bin/env python3
"""Tests for fabric_export.py — the shared GGUF export library."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure fabric_export is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))

from fabric_export import (
    fold_bn,
    merge_lstm_bias,
    ensure_contiguous,
    ModelExporter,
    export_conv_bn,
    export_conv,
    export_linear,
    export_lstm,
)


class TestFoldBN(unittest.TestCase):
    """Test BatchNorm folding into Conv weights."""

    def test_fold_bn_conv2d(self):
        """BN folding for standard Conv2d (scale dim 0)."""
        # Conv weight [OC=2, IC=1, KH=1, KW=1]
        conv_w = np.array([[[[3.0]]], [[[5.0]]]])
        conv_b = np.array([0.0, 0.0])
        bn_w = np.array([2.0, 1.0])      # gamma
        bn_b = np.array([0.5, -0.5])     # beta
        bn_m = np.array([0.0, 0.0])      # running_mean
        bn_v = np.array([1.0, 1.0])      # running_var (scale = gamma/sqrt(var+eps) ≈ gamma)

        w, b = fold_bn(conv_w, conv_b, bn_w, bn_b, bn_m, bn_v, eps=0.0)

        # w[0] = 3.0 * 2.0 = 6.0,  w[1] = 5.0 * 1.0 = 5.0
        np.testing.assert_allclose(w[0, 0, 0, 0], 6.0, atol=1e-6)
        np.testing.assert_allclose(w[1, 0, 0, 0], 5.0, atol=1e-6)

        # b = beta + (conv_b - mean) * scale = [0.5, -0.5] + 0 = [0.5, -0.5]
        np.testing.assert_allclose(b, [0.5, -0.5], atol=1e-6)

    def test_fold_bn_conv2d_no_bias(self):
        """BN folding when conv has no bias."""
        conv_w = np.ones((2, 1, 1, 1), dtype=np.float32)
        bn_w = np.array([1.0, 1.0])
        bn_b = np.array([0.0, 0.0])
        bn_m = np.array([1.0, 2.0])   # non-zero means
        bn_v = np.array([1.0, 1.0])

        w, b = fold_bn(conv_w, None, bn_w, bn_b, bn_m, bn_v, eps=0.0)

        # b = beta - mean * scale = [0, 0] - [1, 2] * [1, 1] = [-1, -2]
        np.testing.assert_allclose(b, [-1.0, -2.0], atol=1e-6)

    def test_fold_bn_deconv(self):
        """BN folding for ConvTranspose2d (scale dim 1)."""
        # [IC=1, OC=2, KH=1, KW=1]
        conv_w = np.array([[[[1.0]], [[2.0]]]])
        bn_w = np.array([3.0, 4.0])      # scales OC dim
        bn_b = np.array([0.0, 0.0])
        bn_m = np.array([0.0, 0.0])
        bn_v = np.array([1.0, 1.0])

        w, b = fold_bn(conv_w, None, bn_w, bn_b, bn_m, bn_v, eps=0.0, is_deconv=True)

        # w[:, 0] scaled by 3.0, w[:, 1] scaled by 4.0
        np.testing.assert_allclose(w[0, 0, 0, 0], 3.0, atol=1e-6)
        np.testing.assert_allclose(w[0, 1, 0, 0], 8.0, atol=1e-6)

    def test_fold_bn_with_eps(self):
        """BN folding with non-zero eps."""
        conv_w = np.ones((1, 1, 1, 1), dtype=np.float32)
        bn_w = np.array([1.0])
        bn_b = np.array([0.0])
        bn_m = np.array([0.0])
        bn_v = np.array([0.0])   # var=0 -> scale = 1/sqrt(eps)

        w, b = fold_bn(conv_w, None, bn_w, bn_b, bn_m, bn_v, eps=1e-5)

        expected_scale = 1.0 / np.sqrt(1e-5)
        np.testing.assert_allclose(w[0, 0, 0, 0], expected_scale, rtol=1e-5)


class TestMergeLSTMBias(unittest.TestCase):
    def test_merge(self):
        b_ih = np.array([1.0, 2.0, 3.0, 4.0])
        b_hh = np.array([0.5, 0.5, 0.5, 0.5])
        result = merge_lstm_bias(b_ih, b_hh)
        np.testing.assert_allclose(result, [1.5, 2.5, 3.5, 4.5])


class TestEnsureContiguous(unittest.TestCase):
    def test_already_contiguous(self):
        a = np.array([1.0, 2.0, 3.0])
        result = ensure_contiguous(a)
        assert result.flags['C_CONTIGUOUS']

    def test_non_contiguous(self):
        a = np.array([[1, 2], [3, 4], [5, 6]])
        sliced = a[:, 0]  # non-contiguous view
        assert not sliced.flags['C_CONTIGUOUS']
        result = ensure_contiguous(sliced)
        assert result.flags['C_CONTIGUOUS']
        np.testing.assert_array_equal(result, [1, 3, 5])


class TestModelExporter(unittest.TestCase):
    def test_type_selection(self):
        """Weights (2D+) should be F16, biases (1D) should be F32."""
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            exporter = ModelExporter(path, "test")

            # 2D weight -> should become F16
            w = np.ones((3, 2), dtype=np.float32)
            exporter.add_tensor("weight", w)
            assert exporter.tensors["weight"].dtype == np.float16

            # 1D bias -> should stay F32
            b = np.ones((3,), dtype=np.float32)
            exporter.add_tensor("bias", b)
            assert exporter.tensors["bias"].dtype == np.float32

            # 4D conv weight -> F16
            cw = np.ones((2, 1, 3, 3), dtype=np.float32)
            exporter.add_tensor("conv.weight", cw)
            assert exporter.tensors["conv.weight"].dtype == np.float16

            exporter.write()
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_roundtrip(self):
        """Write and read back tensors via gguf."""
        import gguf

        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            exporter = ModelExporter(path, "roundtrip-test")
            exporter.add_tensor("x", np.array([1.0, 2.0, 3.0], dtype=np.float32))
            exporter.write()

            # Read back
            reader = gguf.GGUFReader(path)
            tensor_names = [t.name for t in reader.tensors]
            assert "x" in tensor_names
        finally:
            os.unlink(path)


class MockStateDict(dict):
    """Simulates a PyTorch state_dict with .numpy() on tensors."""

    class FakeTensor:
        def __init__(self, data):
            self._data = np.asarray(data, dtype=np.float32)
            self.ndim = self._data.ndim

        def numpy(self):
            return self._data

    def __setitem__(self, key, value):
        super().__setitem__(key, self.FakeTensor(value))

    def get(self, key, default=None):
        v = super().get(key, default)
        return v


class TestExportConvBN(unittest.TestCase):
    def test_export_conv_bn(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            sd = MockStateDict()
            sd["conv.weight"] = np.ones((4, 3, 3, 3))
            sd["bn.weight"] = np.ones(4)
            sd["bn.bias"] = np.zeros(4)
            sd["bn.running_mean"] = np.zeros(4)
            sd["bn.running_var"] = np.ones(4)

            exporter = ModelExporter(path, "test")
            export_conv_bn(exporter, sd, "conv", "bn", "out_conv")

            assert "out_conv.weight" in exporter.tensors
            assert "out_conv.bias" in exporter.tensors
            assert exporter.tensors["out_conv.weight"].dtype == np.float16  # 4D -> F16
            assert exporter.tensors["out_conv.bias"].dtype == np.float32   # 1D -> F32
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestExportConv(unittest.TestCase):
    def test_export_conv_with_bias(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            sd = MockStateDict()
            sd["conv.weight"] = np.ones((4, 3, 3, 3))
            sd["conv.bias"] = np.zeros(4)

            exporter = ModelExporter(path, "test")
            export_conv(exporter, sd, "conv", "out")

            assert "out.weight" in exporter.tensors
            assert "out.bias" in exporter.tensors
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_conv_no_bias(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            sd = MockStateDict()
            sd["conv.weight"] = np.ones((4, 3, 3, 3))

            exporter = ModelExporter(path, "test")
            export_conv(exporter, sd, "conv", "out")

            assert "out.weight" in exporter.tensors
            assert "out.bias" not in exporter.tensors
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestExportLinear(unittest.TestCase):
    def test_export_linear(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            sd = MockStateDict()
            sd["fc.weight"] = np.ones((10, 20))
            sd["fc.bias"] = np.zeros(10)

            exporter = ModelExporter(path, "test")
            export_linear(exporter, sd, "fc", "output_fc")

            assert "output_fc.weight" in exporter.tensors
            assert "output_fc.bias" in exporter.tensors
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestExportLSTM(unittest.TestCase):
    def test_export_lstm_bidirectional_2layers(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            sd = MockStateDict()
            hidden = 128
            input_size = 64

            for layer in [0, 1]:
                for direction in ["", "_reverse"]:
                    suffix = f"_l{layer}{direction}"
                    sd[f"lstm.weight_ih{suffix}"] = np.ones((4 * hidden, input_size if layer == 0 else 2 * hidden))
                    sd[f"lstm.weight_hh{suffix}"] = np.ones((4 * hidden, hidden))
                    sd[f"lstm.bias_ih{suffix}"] = np.ones(4 * hidden) * 0.1
                    sd[f"lstm.bias_hh{suffix}"] = np.ones(4 * hidden) * 0.2

            exporter = ModelExporter(path, "test")
            export_lstm(exporter, sd, "lstm", "enc", num_layers=2, bidirectional=True)

            # Should have 3 tensors per direction per layer = 3 * 4 = 12
            lstm_tensors = [k for k in exporter.tensors if k.startswith("enc.")]
            assert len(lstm_tensors) == 12, f"Expected 12 LSTM tensors, got {len(lstm_tensors)}"

            # Check bias merge: 0.1 + 0.2 = 0.3
            bias = exporter.tensors["enc.bias_l0"]
            np.testing.assert_allclose(bias, np.ones(4 * hidden) * 0.3, atol=1e-6)

            # Forward direction should NOT have "_reverse" suffix
            assert "enc.weight_ih_l0" in exporter.tensors
            assert "enc.weight_hh_l0" in exporter.tensors
            assert "enc.bias_l0" in exporter.tensors

            # Reverse direction
            assert "enc.weight_ih_l0_reverse" in exporter.tensors
            assert "enc.weight_hh_l0_reverse" in exporter.tensors
            assert "enc.bias_l0_reverse" in exporter.tensors
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_export_lstm_unidirectional(self):
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            path = f.name

        try:
            sd = MockStateDict()
            sd["lstm.weight_ih_l0"] = np.ones((16, 8))
            sd["lstm.weight_hh_l0"] = np.ones((16, 4))
            sd["lstm.bias_ih_l0"] = np.ones(16)
            sd["lstm.bias_hh_l0"] = np.ones(16)

            exporter = ModelExporter(path, "test")
            export_lstm(exporter, sd, "lstm", "enc", num_layers=1, bidirectional=False)

            lstm_tensors = [k for k in exporter.tensors if k.startswith("enc.")]
            assert len(lstm_tensors) == 3, f"Expected 3 LSTM tensors, got {len(lstm_tensors)}"
            assert "enc.weight_ih_l0" in exporter.tensors
            assert "enc.bias_l0" in exporter.tensors
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
