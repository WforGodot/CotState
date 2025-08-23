"""
Ensures torch/torchaudio/torchvision are installed (any version/CUDA).
Then force-reinstalls numpy and the rest of the stack.

- Leaves existing torch stack alone if present.
- Installs any missing torch components from PyPI (no CUDA/index pin).
- Force-reinstalls: numpy, transformers, accelerate, datasets, scikit-learn,
  matplotlib, ipywidgets, tqdm, bitsandbytes, transformer_lens.
- Saves manifest to outputs/install/manifest.json.
- Overwrites pip log and manifest on re-runs.
"""
import json
import subprocess
import sys
from pathlib import Path
import importlib

# Pinned non-PyTorch stack (as in your original script)
PINNED_PKGS = [
    "numpy",  # will be force-reinstalled (latest available unless pinned here)
    "transformers==4.55.2",
    "accelerate==1.10.0",
    "datasets==4.0.0",
    "scikit-learn==1.7.1",
    "matplotlib==3.10.5",
    "ipywidgets==8.1.7",
    "tqdm==4.67.1",
    "bitsandbytes==0.46.1",
    "transformer_lens==2.16.1",
]

def pip_install(args, log_path: Path):
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--log", str(log_path)] + args
    print("pip install", " ".join(args))
    subprocess.check_call(cmd)

def mod_version(name):
    try:
        m = __import__(name)
        return getattr(m, "__version__", None)
    except Exception:
        return None

def ensure_pt_stack(log_file: Path):
    """Ensure torch, torchaudio, torchvision are importable (any version/CUDA)."""
    needed = []
    for name in ["torch", "torchaudio", "torchvision"]:
        v = mod_version(name)
        if v is None:
            needed.append(name)
    if needed:
        print("PyTorch components missing:", ", ".join(needed))
        # No index override; let pip resolve appropriate wheels (CPU/GPU)
        pip_install(needed, log_file)
    else:
        print("PyTorch stack present; skipping install.")

def force_reinstall_rest(log_file: Path):
    """Force-reinstall numpy and the rest of the stack (pins preserved above)."""
    # numpy first to avoid transient ABI weirdness with downstream libs
    numpy_args = ["--upgrade", "--force-reinstall", "numpy"]
    pip_install(numpy_args, log_file)

    others = [p for p in PINNED_PKGS if not p.startswith("numpy")]
    if others:
        pip_install(["--upgrade", "--force-reinstall"] + others, log_file)

def write_manifest(out_dir: Path):
    importlib.invalidate_caches()
    def safe_ver(name):
        try:
            m = __import__(name)
            return getattr(m, "__version__", None)
        except Exception:
            return None

    # torch cuda version (best-effort, may be None)
    torch_cuda = None
    try:
        import torch
        torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        pass

    manifest = {
        "torch": safe_ver("torch"),
        "torch_cuda": torch_cuda,
        "torchaudio": safe_ver("torchaudio"),
        "torchvision": safe_ver("torchvision"),
        "numpy": safe_ver("numpy"),
        "transformers": safe_ver("transformers"),
        "accelerate": safe_ver("accelerate"),
        "datasets": safe_ver("datasets"),
        "scikit_learn": safe_ver("sklearn"),
        "matplotlib": safe_ver("matplotlib"),
        "ipywidgets": safe_ver("ipywidgets"),
        "tqdm": safe_ver("tqdm"),
        "bitsandbytes": safe_ver("bitsandbytes"),
        "transformer_lens": safe_ver("transformer_lens"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def main():
    out_dir = Path("outputs/install")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "pip.log"

    # Fresh log each run
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass

    # 1) Make sure torch stack exists (any version/CUDA)
    ensure_pt_stack(log_file)

    # 2) Force-reinstall numpy and the rest
    force_reinstall_rest(log_file)

    # 3) Manifest
    write_manifest(out_dir)

    print("✅ Done. Manifest written to outputs/install/manifest.json")
    print(f"📄 pip log: {log_file}")

if __name__ == "__main__":
    main()
