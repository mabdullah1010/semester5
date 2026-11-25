"""M0: check imports, record environment to results/m0/env.json."""
import importlib, json, platform, os, sys, time
from importlib import metadata
from pathlib import Path

PKGS = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "soundfile", "librosa", "opensmile",
        "torch", "torchaudio", "transformers", "krippendorff", "statsmodels", "matplotlib",
        "tqdm", "pyyaml", "pytest", "lightgbm"]
MODS = ["opensmile", "torch", "transformers", "sklearn", "soundfile", "librosa", "krippendorff"]


def main():
    info = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "versions": {}, "imports_ok": {}}
    for p in PKGS:
        try:
            info["versions"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["versions"][p] = None
    for m in MODS:
        try:
            importlib.import_module(m); info["imports_ok"][m] = True
        except Exception as e:
            info["imports_ok"][m] = f"FAIL: {e!r}"
    import torch
    info["gpu_available"] = bool(torch.cuda.is_available())
    info["gpu_name"] = torch.cuda.get_device_name(0) if info["gpu_available"] else None
    info["torch_threads"] = torch.get_num_threads()
    with open("/proc/meminfo") as f:
        info["mem_total_kb"] = int(f.readline().split()[1])
    info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Path("results/m0").mkdir(parents=True, exist_ok=True)
    Path("results/m0/env.json").write_text(json.dumps(info, indent=2))
    ok = all(v is True for v in info["imports_ok"].values())
    print("imports:", info["imports_ok"]); print("GPU:", info["gpu_available"])
    print("M0 gate:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
