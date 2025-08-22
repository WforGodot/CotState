#!/usr/bin/env python3
"""
Install PyTorch 2.8.0 matching the local CUDA runtime if available.

Behavior
- If torch==2.8.0 is already installed, no action is taken.
- Otherwise, detect CUDA version (nvidia-smi, nvcc, or env), map to a
  supported channel (cu124 | cu121 | cu118 | cpu), and install via pip.

Notes
- This script prefers the modern per-channel index URLs used by PyTorch.
- If the first attempt fails, it falls back to alternative commands.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from typing import Optional, Tuple


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def have_torch_2_8() -> bool:
    try:
        import torch  # type: ignore

        ver = getattr(torch, "__version__", "")
        return ver.startswith("2.8.0")
    except Exception:
        return False


def parse_version_tuple(s: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d+)\.(\d+)", s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def detect_cuda_version() -> Optional[Tuple[int, int]]:
    # 1) Explicit env hint (e.g., CUDA_VERSION=12.1)
    env = os.environ.get("CUDA_VERSION") or os.environ.get("CUDA_HOME_VERSION")
    if env:
        vt = parse_version_tuple(env)
        if vt:
            return vt

    # 2) nvidia-smi (works on most driver installs)
    try:
        p = run(["nvidia-smi"])
        m = re.search(r"CUDA Version\s*:\s*(\d+\.\d+)", p.stdout)
        if m:
            vt = parse_version_tuple(m.group(1))
            if vt:
                return vt
    except FileNotFoundError:
        pass

    # 3) nvcc --version (developer toolkit)
    try:
        p = run(["nvcc", "--version"])  # type: ignore
        m = re.search(r"release\s*(\d+\.\d+)", p.stdout)
        if m:
            vt = parse_version_tuple(m.group(1))
            if vt:
                return vt
    except FileNotFoundError:
        pass

    # 4) Last resort: if an older torch exists, read torch.version.cuda
    try:
        import torch  # type: ignore

        cv = getattr(getattr(torch, "version", object()), "cuda", None)
        if isinstance(cv, str):
            vt = parse_version_tuple(cv)
            if vt:
                return vt
    except Exception:
        pass

    return None


def map_cuda_to_channel(vt: Optional[Tuple[int, int]]) -> str:
    """Map a CUDA version tuple to a PyTorch wheel channel.

    Known channels (recent releases): cu124, cu121, cu118, cpu
    """
    if vt is None:
        return "cpu"
    major, minor = vt
    if major >= 12 and minor >= 4:
        return "cu124"
    if major >= 12 and minor >= 1:
        return "cu121"
    if major == 11 and minor >= 8:
        return "cu118"
    return "cpu"


def pip_install_torch(channel: str) -> bool:
    py = sys.executable

    if channel == "cpu":
        attempts = [
            [py, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.8.0"],
            [py, "-m", "pip", "install", "torch==2.8.0"],
        ]
    else:
        # Prefer index-url form
        attempts = [
            [py, "-m", "pip", "install", "--index-url", f"https://download.pytorch.org/whl/{channel}", "torch==2.8.0"],
            # Fallbacks: explicit wheel tag with -f finder
            [py, "-m", "pip", "install", f"torch==2.8.0+{channel}", "-f", "https://download.pytorch.org/whl/torch_stable.html"],
            # Extra-index form
            [py, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/torch_stable.html", f"torch==2.8.0+{channel}"],
        ]

    for cmd in attempts:
        print(f"[install_deps] Trying: {' '.join(shlex.quote(c) for c in cmd)}")
        p = subprocess.run(cmd)
        if p.returncode == 0 and have_torch_2_8():
            print("[install_deps] Installed torch==2.8.0 successfully.")
            return True
        else:
            print(f"[install_deps] Attempt failed with code {p.returncode}.")
    return False


def main() -> None:
    if have_torch_2_8():
        print("[install_deps] torch==2.8.0 already installed. Nothing to do.")
        return

    vt = detect_cuda_version()
    chan = map_cuda_to_channel(vt)
    vt_str = f"{vt[0]}.{vt[1]}" if vt else "none"
    print(f"[install_deps] Detected CUDA: {vt_str} -> channel {chan}")

    ok = pip_install_torch(chan)
    if not ok and chan != "cpu":
        print("[install_deps] Falling back to CPU build...")
        ok = pip_install_torch("cpu")

    if not ok:
        print("[install_deps] ERROR: Could not install torch==2.8.0 for your environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()

