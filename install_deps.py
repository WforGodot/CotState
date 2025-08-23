"""
Install policy:
1) Ensure torch/torchaudio/torchvision are present (any version/CUDA).
   - If torch exists but audio/vision are missing, install versions matching torch.
2) Reinstall NumPy to the CURRENT installed version (no upgrade).
3) Reinstall the rest of the stack WITHOUT --upgrade (so torch isn't touched).
   - Use --upgrade-strategy only-if-needed to avoid dependency churn.
Extras:
- --fast: only install what's missing; skip forced reinstalls.
- --skip-datasets: omit datasets (saves pyarrow/pandas time/space).
- --skip-bnb: omit bitsandbytes.
- --use-cache: keep pip cache (default is --no-cache-dir).
Writes manifest to outputs/install/manifest.json and a pip log.
"""
import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

# Base pins (you can tweak here)
BASE_PKGS = [
    "transformers==4.55.2",
    "accelerate==1.10.0",     # depends on torch>=2.0.0; we avoid upgrades
    "datasets==4.0.0",        # heavy due to pyarrow + pandas
    "scikit-learn==1.7.1",
    "matplotlib==3.10.5",
    "ipywidgets==8.1.7",
    "tqdm==4.67.1",
    "bitsandbytes==0.46.1",
    "transformer_lens==2.16.1",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="Only install missing packages; do not force-reinstall.")
    p.add_argument("--skip-datasets", action="store_true",
                   help="Skip 'datasets' (avoids pulling pyarrow/pandas).")
    p.add_argument("--skip-bnb", action="store_true",
                   help="Skip bitsandbytes.")
    p.add_argument("--use-cache", action="store_true",
                   help="Keep pip cache (omit --no-cache-dir).")
    return p.parse_args()

def pip_install(args, log_path: Path, use_cache: bool):
    cmd = [sys.executable, "-m", "pip", "install"]
    if not use_cache:
        cmd += ["--no-cache-dir"]
    cmd += ["--log", str(log_path)]
    cmd += args
    print("pip install", " ".join(args))
    subprocess.check_call(cmd)

def mod_version(name):
    try:
        m = __import__(name)
        return getattr(m, "__version__", None), m
    except Exception:
        return None, None

def ensure_pt_stack(log_file: Path, use_cache: bool):
    """Ensure torch/torchaudio/torchvision import; if torch exists, match versions."""
    torch_ver, torch_mod = mod_version("torch")
    if torch_ver is None:
        # Nothing installed: install torch first (let pip pick correct wheel)
        pip_install(["torch"], log_file, use_cache)
        torch_ver, torch_mod = mod_version("torch")

    # For audio/vision: if missing, install exact same version as torch to avoid ABI mismatch
    audio_ver, _ = mod_version("torchaudio")
    vision_ver, _ = mod_version("torchvision")
    to_install = []
    if audio_ver is None and torch_ver:
        to_install.append(f"torchaudio=={torch_ver}")
    if vision_ver is None and torch_ver:
        to_install.append(f"torchvision=={_match_torchvision_version(torch_ver)}")

    if to_install:
        pip_install(to_install, log_file, use_cache)
    print("PyTorch stack present; skipping torch upgrade.")

def _match_torchvision_version(torch_ver: str) -> str:
    """
    Simple heuristic: torchvision patch level may differ; try matching major.minor.
    Falls back to exact torch_ver if needed.
    """
    parts = torch_ver.split("+")[0].split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.*"
    return torch_ver

def reinstall_numpy_same_version(log_file: Path, use_cache: bool):
    """Reinstall NumPy to the currently installed version (no upgrade)."""
    try:
        import numpy as _np
        pin = f"numpy=={_np.__version__}"
    except Exception:
        pin = "numpy"  # not installed yet; install latest
    pip_install(["--force-reinstall", "--upgrade-strategy", "only-if-needed", pin],
                log_file, use_cache)

def install_rest(log_file: Path, use_cache: bool, fast: bool,
                 skip_datasets: bool, skip_bnb: bool):
    pkgs = list(BASE_PKGS)
    if skip_datasets:
        pkgs = [p for p in pkgs if not p.startswith("datasets")]
    if skip_bnb:
        pkgs = [p for p in pkgs if not p.startswith("bitsandbytes")]

    if fast:
        # Fast path: no force reinstall; pip will skip already satisfied packages.
        pip_install(pkgs, log_file, use_cache)
    else:
        # Reinstall *just these* packages without upgrading their dependencies.
        pip_install(["--force-reinstall", "--upgrade-strategy", "only-if-needed"] + pkgs,
                    log_file, use_cache)

def write_manifest(out_dir: Path):
    importlib.invalidate_caches()
    def safe_ver(name):
        try:
            m = __import__(name)
            return getattr(m, "__version__", None)
        except Exception:
            return None

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
    args = parse_args()

    out_dir = Path("outputs/install")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "pip.log"
    if log_file.exists():
        try: log_file.unlink()
        except Exception: pass

    # 1) Make sure torch stack exists (any version), do NOT upgrade torch
    ensure_pt_stack(log_file, use_cache=args.use_cache)

    # 2) Reinstall NumPy to current version (no upgrade)
    reinstall_numpy_same_version(log_file, use_cache=args.use_cache)

    # 3) (Re)install the rest without upgrading dependencies (so torch stays put)
    install_rest(
        log_file,
        use_cache=args.use_cache,
        fast=args.fast,
        skip_datasets=args.skip_datasets,
        skip_bnb=args.skip_bnb,
    )

    write_manifest(out_dir)
    print("✅ Done. Manifest written to outputs/install/manifest.json")
    print(f"📄 pip log: {log_file}")

if __name__ == "__main__":
    main()
