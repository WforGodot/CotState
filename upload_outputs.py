#!/usr/bin/env python3
"""
Zip a selected file or folder (defaults to src/cot/outputs/probes) and upload to Google Drive via rclone.

Features:
- Downloads a local rclone binary if not installed (no sudo/admin needed).
- Creates its own rclone.conf next to this script and stores tokens there.
- Non-interactive setup where possible; on first run, opens browser for OAuth.
- Supports --folder-id to upload into a specific Google Drive folder.
- Optional --make-link to request a shareable link for the uploaded file.

 Usage:
  python upload_outputs.py [--path FILE_OR_DIR]
                           [--outputs-path DIR]
                                           [--folder-id FOLDER_ID]
                                           [--remote gdrive]
                                           [--dest REMOTE_PATH]
                                           [--make-link]
                                           [--keep-zip]

Defaults:
  path: <script_dir>/src/cot/outputs/probes
  outputs-path (deprecated default base): <script_dir>/src/cot/outputs/probes
  remote: gdrive
  rclone.conf: <script_dir>/rclone.conf
  rclone bin:  <script_dir>/.rclone-bin/<platform>/rclone[.exe]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

# ------------------------- Utility -------------------------

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def run(cmd, env=None, check=True, capture_output=False):
    kw = dict(env=env)
    if capture_output:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.STDOUT
        res = subprocess.run(cmd, **kw, text=True)
        if check and res.returncode != 0:
            raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}")
        return res.stdout
    else:
        subprocess.check_call(cmd, **kw)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def zip_any(src: Path, out_zip: Path) -> Path:
    """Zip a file or a directory into out_zip and return out_zip."""
    src = src.resolve()
    out_zip = out_zip.resolve()
    ensure_dir(out_zip)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(src))
        else:
            # Single file
            zf.write(src, arcname=src.name)
    return out_zip

# ------------------------- Rclone bootstrap -------------------------

def detect_platform_triplet():
    sysname = platform.system().lower()        # 'linux', 'darwin', 'windows'
    machine = platform.machine().lower()       # 'x86_64', 'amd64', 'arm64', 'aarch64', 'armv7l', ...
    # Normalize arch
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("armv7l", "armv7"):
        # rclone provides linux-arm-v7 only on Linux
        arch = "arm-v7"
    elif machine in ("armv6l", "armv6"):
        arch = "arm-v6"
    else:
        raise SystemExit(f"Unsupported CPU arch for auto-install: {machine}")
    # Map to rclone triplet
    if sysname == "linux":
        if arch in ("arm-v7", "arm-v6"):
            triplet = f"linux-{arch}"
        else:
            triplet = f"linux-{arch}"
    elif sysname == "darwin":
        if arch not in ("amd64", "arm64"):
            raise SystemExit(f"Unsupported macOS arch for auto-install: {machine}")
        triplet = f"darwin-{arch}"
    elif sysname == "windows":
        if arch not in ("amd64", "arm64"):
            # Windows arm64 is supported; others are not common
            raise SystemExit(f"Unsupported Windows arch for auto-install: {machine}")
        triplet = f"windows-{arch}"
    else:
        raise SystemExit(f"Unsupported OS for auto-install: {sysname}")
    return sysname, triplet

def download_rclone_if_needed(bin_dir: Path) -> Path:
    """
    Returns absolute path to rclone binary (local vendored copy if needed).
    """
    # If rclone already in PATH, use it.
    rclone_in_path = shutil.which("rclone")
    if rclone_in_path:
        return Path(rclone_in_path)

    sysname, triplet = detect_platform_triplet()
    bin_dir = bin_dir.resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)

    exe_name = "rclone.exe" if sysname == "windows" else "rclone"
    local_bin = bin_dir / exe_name
    if local_bin.exists():
        return local_bin

    # Download "current" build zip from official site
    url = f"https://downloads.rclone.org/rclone-current-{triplet}.zip"
    print(f"[INFO] rclone not found in PATH. Downloading: {url}")
    try:
        with urlopen(url) as resp:
            data = resp.read()
    except (URLError, HTTPError) as e:
        raise SystemExit(f"Failed to download rclone: {e}")

    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "rclone.zip"
        zip_path.write_bytes(data)
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only the rclone binary from the nested folder
            members = zf.namelist()
            # Find entry that ends with /rclone(.exe)
            target = None
            for m in members:
                base = m.split("/")[-1]
                if base == exe_name:
                    target = m
                    break
            if not target:
                raise SystemExit("Downloaded archive didn't contain rclone binary.")
            zf.extract(target, td)
            extracted = Path(td) / target
            shutil.copy2(extracted, local_bin)

    if sysname != "windows":
        local_bin.chmod(0o755)

    print(f"[INFO] rclone bootstrapped at: {local_bin}")
    return local_bin

def ensure_remote(rclone_bin: Path, conf_path: Path, remote_name: str):
    env = os.environ.copy()
    env["RCLONE_CONFIG"] = str(conf_path)

    # List existing remotes
    try:
        out = run([str(rclone_bin), "listremotes"], env=env, capture_output=True)
        remotes = {r.strip().rstrip(":") for r in out.splitlines() if r.strip()}
    except Exception:
        remotes = set()

    if remote_name in remotes:
        return  # already configured

    print(f"[INFO] Creating rclone remote '{remote_name}' for Google Drive (scope=drive.file)")
    # Non-interactive creation; will try opening browser automatically.
    # If this fails (headless), we fall back to interactive "rclone config".
    create_cmd = [
        str(rclone_bin), "--config", str(conf_path),
        "config", "create", remote_name, "drive",
        "scope", "drive.file",
        "config_is_local", "true",
    ]
    try:
        run(create_cmd)
    except Exception as e:
        eprint("[WARN] Non-interactive create failed, starting interactive rclone config...")
        run([str(rclone_bin), "--config", str(conf_path), "config"])

# ------------------------- Main flow -------------------------

def main():
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Zip and upload a file or directory to Google Drive via rclone")
    parser.add_argument(
        "--path",
        default=str(script_dir / "src" / "cot" / "outputs" / "probes"),
        help="Path to a file or directory to zip and upload (default: src/cot/outputs/probes)",
    )
    # Backward-compat: keep --outputs-path; if provided, it will be used when --path is not explicitly set.
    parser.add_argument(
        "--outputs-path",
        default=None,
        help="Deprecated: path to outputs directory. Use --path instead.",
    )
    parser.add_argument("--folder-id", default=None, help="Google Drive folder ID (optional)")
    parser.add_argument("--remote", default="gdrive", help="rclone remote name (default: gdrive)")
    parser.add_argument("--dest", default=None,
                        help="Destination like 'gdrive:path/in/drive/file.zip'. If given, ignores --folder-id.")
    parser.add_argument("--rclone-conf", default=str(script_dir / "rclone.conf"),
                        help="Path to rclone config file used by this script")
    parser.add_argument("--make-link", action="store_true",
                        help="Attempt to create and print a shareable link after upload")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the created zip instead of deleting after upload")
    args = parser.parse_args()

    # Determine source to zip: prefer --path; else fallback to --outputs-path if provided for compatibility
    src_path_str = args.path if args.path is not None else args.outputs_path
    src_path = Path(src_path_str)
    if not src_path.exists():
        raise SystemExit(f"Path not found: {src_path}")

    # 1) Ensure rclone binary available (vendored if needed)
    rclone_bin = download_rclone_if_needed(script_dir / ".rclone-bin")

    # 2) Ensure remote configured (opens browser on first run)
    conf_path = Path(args.rclone_conf)
    ensure_dir(conf_path)
    ensure_remote(rclone_bin, conf_path, args.remote)

    # 3) Zip selected path (file or directory)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = src_path.parent
    zip_path = base_dir / f"outputs_{ts}.zip"
    print(f"[INFO] Zipping {src_path} -> {zip_path}")
    zip_path = zip_any(src_path, zip_path)

    # 4) Build upload command
    env = os.environ.copy()
    env["RCLONE_CONFIG"] = str(conf_path)

    if args.dest:
        dest = args.dest
        extra = []
    else:
        dest = f"{args.remote}:{zip_path.name}"
        extra = []
        if args.folder_id:
            extra += ["--drive-root-folder-id", args.folder_id]

    print(f"[INFO] Uploading {zip_path.name} ...")
    run([str(rclone_bin), "copyto", str(zip_path), dest, "--progress"] + extra, env=env)

    # 5) Optional link
    if args.make_link:
        try:
            link_cmd = [str(rclone_bin), "link", dest]
            if args.folder_id and not args.dest:
                link_cmd += ["--drive-root-folder-id", args.folder_id]
            link = run(link_cmd, env=env, capture_output=True).strip()
            if link:
                print(f"[INFO] Shareable link:\n{link}")
            else:
                print("[INFO] Link not available (permissions or policy).")
        except Exception as e:
            eprint(f"[WARN] Couldn't create link: {e}")

    # 6) Clean up
    if not args.keep_zip:
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("[DONE] Upload complete.")

if __name__ == "__main__":
    main()
