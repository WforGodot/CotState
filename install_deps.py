"""
Installs all required libraries for this project, including:
- torch==2.8.0+cu126, torchaudio==2.8.0+cu126, torchvision==0.23.0+cu126
- transformers==4.55.2, accelerate==1.10.0, datasets==4.0.0
- scikit-learn==1.7.1, matplotlib==3.10.5, ipywidgets==8.1.7, tqdm==4.67.1
- bitsandbytes==0.46.1
- transformer_lens==2.16.1

Behavior:
- Does NOT reinstall torch/torchaudio/torchvision if already at the correct versions.
- Reinstalls NumPy ONLY if a PyTorch component was (re)installed.
- Saves a manifest of installed versions to outputs/install/manifest.json.
- Re-runs overwrite the manifest and logs.
"""
import json
import subprocess
import sys
from pathlib import Path

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
DESIRED = {
    "torch": "2.8.0+cu126",
    "torchaudio": "2.8.0+cu126",
    "torchvision": "0.23.0+cu126",
}
DESIRED_CUDA = "12.6"  # torch.version.cuda should report this for cu126 wheels


def pip_install(args, log_path: Path):
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--log", str(log_path)] + args
    print("pip install", " ".join(args))
    subprocess.check_call(cmd)


def get_mod_version(mod_name):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", None), mod
    except Exception:
        return None, None


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

    # Detect current torch/cuda
    torch_ver, torch_mod = get_mod_version("torch")
    cur_cuda = getattr(getattr(torch_mod, "version", None), "cuda", None) if torch_mod else None
    print(f"Detected Torch: {torch_ver} | CUDA: {cur_cuda}")

    # Figure out which PyTorch components (if any) need installing/updating
    to_install_pt = []
    need_torch = not (torch_ver == DESIRED["torch"] and cur_cuda == DESIRED_CUDA)
    if need_torch:
        to_install_pt.append(f"torch=={DESIRED['torch']}")
    ta_ver, _ = get_mod_version("torchaudio")
    if ta_ver != DESIRED["torchaudio"]:
        to_install_pt.append(f"torchaudio=={DESIRED['torchaudio']}")
    tv_ver, _ = get_mod_version("torchvision")
    if tv_ver != DESIRED["torchvision"]:
        to_install_pt.append(f"torchvision=={DESIRED['torchvision']}")

    pytorch_changed = False
    if to_install_pt:
        print("Installing/aligning PyTorch stack:", ", ".join(to_install_pt))
        pip_install(["--index-url", PYTORCH_INDEX_URL] + to_install_pt, log_file)
        pytorch_changed = True
    else:
        print("PyTorch stack already at desired versions; skipping reinstall.")

    # Reinstall current NumPy ONLY if PyTorch stack changed (helps ABI edges after torch swaps)
    if pytorch_changed:
        try:
            import numpy as _np  # noqa
            numpy_pin = f"numpy=={_np.__version__}"
            print(f"Reinstalling {numpy_pin} to avoid ABI mismatch...")
            pip_install(["--force-reinstall", numpy_pin], log_file)
        except Exception:
            print("NumPy not detectable pre-reinstall; installing latest stable NumPy from PyPI.")
            pip_install(["numpy"], log_file)

    # Core stack (pinned to latest stable compatible with Torch 2.8)
    pkgs = [
        "transformers==4.55.2",
        "accelerate==1.10.0",
        "datasets==4.0.0",
        "scikit-learn==1.7.1",
        "matplotlib==3.10.5",
        "ipywidgets==8.1.7",
        "tqdm==4.67.1",
        "bitsandbytes==0.46.1",
        "transformer_lens==2.16.1",
        "scikit-learn>=1.2",
    ]
    pip_install(pkgs, log_file)

    # Build manifest (be tolerant if some imports fail)
    def safe_ver(mod_name):
        try:
            mod = __import__(mod_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None

    import importlib
    importlib.invalidate_caches()

    torch_ver, torch_mod = get_mod_version("torch")
    torch_cuda = getattr(getattr(torch_mod, "version", None), "cuda", None) if torch_mod else None

    manifest = {
        "torch": torch_ver,
        "torch_cuda": torch_cuda,
        "torchaudio": safe_ver("torchaudio"),
        "torchvision": safe_ver("torchvision"),
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
    print("✅ Dependencies installed. Manifest written to outputs/install/manifest.json")
    print(f"📄 pip log: {log_file}")


if __name__ == "__main__":
    main()
