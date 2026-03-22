"""
LTX-2.3 Reference Test — Google Colab (GPU runtime)
Uses the inference.py from the LTX-Video repo directly.
Paste in a Colab cell and run.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# 1. Install LTX-Video
if Path("/tmp/ltx-ref").exists():
    shutil.rmtree("/tmp/ltx-ref")
subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/Lightricks/LTX-Video.git", "/tmp/ltx-ref"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "/tmp/ltx-ref"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"])

print("Installed!")

# 2. Run inference
subprocess.check_call(
    [
        sys.executable,
        "/tmp/ltx-ref/inference.py",
        "--ckpt_path",
        "Lightricks/LTX-Video-0.9.7",
        "--prompt",
        "a cat walking on a sunny street",
        "--height",
        "480",
        "--width",
        "704",
        "--num_frames",
        "41",
        "--seed",
        "42",
        "--output_path",
        "/content/ltx_ref_output",
    ]
)

print("Done! Check /content/ltx_ref_output/")
