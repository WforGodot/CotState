"""
Policy:
- Ensure `torch` is installed (any version/CUDA). Do NOT upgrade/downgrade it.
- Force-reinstall NumPy (same version as currently installed).
- For all other libs, install/upgrade ONLY if below a 'reasonable' minimum.
- Use a constraints file to pin the currently-installed torch (so pip won't touch it).
- Write manifest to outputs/install/manifest.json and pip log to outputs/install/pip.log.
"""

import json
import importlib
import subprocess
import sys
from pathlib import Path

from packaging.version import parse as V

# ---- Reasonable minimums (tune as you like; these are conservative) ----
MIN_OK = {
    "transformers":      "4.37.0",
    "accelerate":        "0.26.0",
    "datasets":          "2.14.0",
    "scikit-learn":      "1.1.0",
    "matplotlib":        "3.5.0",
    "ipywidgets":        "8.0.0",
    "tqdm":              "4.64.0",
    "bitsandbytes":      "0.41.0",
    "transformer_lens":  "2.0.0",
}

# If you *also* want to enforce torchaudio/torchvision, set these True.
ENFORCE_TORCHAUDIO = False
ENFORCE_TORCHVISION = False

def pip_install(args, log_path: Path):
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade-strategy", "only-if-needed", "--log", str(log_path)]
    print("pip install", " ".join(args))
    subprocess.check_call(cmd + args)

def mod_version(name):
    try:
        m = __import__(name)
        return getattr(m, "__version__", None)
    except Exception:
        return None

def ensure_torch(log_file: Path):
    """Ensure torch exists (no upgrades)."""
    torch_ver = mod_version("torch")
    if torch_ver is None:
        pip_install(["torch"], log_file)  # let pip pick appropriate wheel
        torch_ver = mod_version("torch")
    return torch_ver

def write_constraints(out_dir: Path):
    """Pin currently-installed torch (+audio/vision if present/enforced)."""
    lines = []
    tv = mod_version("torch")
    if tv:
        lines.append(f"torch=={tv}")
    if ENFORCE_TORCHAUDIO:
        tav = mod_version("torchaudio")
        if tav:
            lines.append(f"torchaudio=={tav}")
    if ENFORCE_TORCHVISION:
        tvv = mod_version("torchvision")
        if tvv:
            lines.append(f"torchvision=={tvv}")
    cpath = out_dir / "constraints.txt"
    cpath.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return cpath

def reinstall_numpy_same_version(log_file: Path):
    """Force-reinstall NumPy to its current version (prevents ABI edge cases)."""
    cur = mod_version("numpy")
    pin = f"numpy=={cur}" if cur else "numpy"
    pip_install(["--force-reinstall", pin], log_file)

def ensure_reasonable_versions(log_file: Path, constraints_path: Path):
    """Install/upgrade ONLY the packages that are missing or below the MIN_OK version."""
    to_fix = []
    for name, min_v in MIN_OK.items():
        cur = mod_version(name)
        if cur is None or V(cur) < V(min_v):
            # Ask for at least min version; let pip resolve deps (but pin torch via -c)
            to_fix.append(f"{name}>={min_v}")

    if not to_fix:
        print("All non-PyTorch libraries already at reasonable versions; skipping.")
        return

    # Important: use constraints to prevent torch upgrades; avoid --upgrade here.
    pip_install(["-c", str(constraints_path)] + to_fix, log_file)

def write_manifest(out_dir: Path):
    importlib.invalidate_caches()
    def safe_ver(n):
        try:
            m = __import__(n)
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
    out_dir = Path("outputs/install")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "pip.log"
    if log_file.exists():
        try: log_file.unlink()
        except Exception: pass

    ensure_torch(log_file)                        # make sure torch exists (no upgrades)
    constraints = write_constraints(out_dir)      # pin current torch so nothing touches it
    reinstall_numpy_same_version(log_file)        # only forced reinstall
    ensure_reasonable_versions(log_file, constraints)  # touch others only if too old/missing
    write_manifest(out_dir)

    print("✅ Done. Manifest written to outputs/install/manifest.json")
    print(f"📄 pip log: {log_file}")

if __name__ == "__main__":
    main()
