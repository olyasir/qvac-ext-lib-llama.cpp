#!/usr/bin/env python3
"""Generate a minimal GGUF file for testing the fabric SDK.

Creates a tiny model with:
  - A 2D weight (3x2) stored as F16
  - A 1D bias (2) stored as F32
  - A depthwise conv weight [3,3,1,4] stored as F32 (to test auto-cast)
  - A transpose conv weight [2,2,3,3] stored as F32 (to test auto-cast)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf

output = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fabric_test.gguf"

writer = gguf.GGUFWriter(output, "fabric-test")

# Linear weight [3, 2] — will be F16
w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)  # [2, 3]
writer.add_tensor("linear.weight", w)

# Linear bias [2] — will be F32
b = np.array([0.1, 0.2], dtype=np.float32)
writer.add_tensor("linear.bias", b)

# Depthwise conv weight [3,3,1,4] in ggml order = [KW=3, KH=3, 1, C=4] — stored F32 to test auto-cast
dw_w = np.ones((4, 1, 3, 3), dtype=np.float32) * 0.5
writer.add_tensor("dw_conv.weight", dw_w)

# Transpose conv weight [IC=2, OC=2, KH=3, KW=3] — stored F32 to test auto-cast
tc_w = np.ones((2, 2, 3, 3), dtype=np.float32) * 0.25
writer.add_tensor("tconv.weight", tc_w)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"Generated {output}")
