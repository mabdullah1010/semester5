"""File listing helpers. Every script must list files through here (Rule 11)."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

JUNK_PREFIXES = ("._",)
JUNK_NAMES = {".DS_Store", "Thumbs.db", "README.txt~"}
JUNK_SUFFIXES = (".pk",)

# Project dirs that live inside the dataset folder and must never be walked as data.
PROJECT_DIRS = {".venv", "src", "tests", "data", "features", "results", "figures",
                "report", ".git", "__pycache__", ".pytest_cache"}


def is_junk(name: str) -> bool:
    return name.startswith(JUNK_PREFIXES) or name in JUNK_NAMES or name.endswith(JUNK_SUFFIXES)


def load_config(path: str | os.PathLike = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def list_files(root, suffix: str | None = None, count_junk: list | None = None):
    """Recursively list non-junk files under `root` (sorted). If `count_junk` is a list,
    skipped junk paths are appended to it. Project dirs are skipped at the top level."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath) == root:
            dirnames[:] = [d for d in dirnames if d not in PROJECT_DIRS and not is_junk(d)]
        else:
            dirnames[:] = [d for d in dirnames if not is_junk(d)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_junk(fn):
                if count_junk is not None:
                    count_junk.append(str(p))
                continue
            if suffix is None or fn.endswith(suffix):
                out.append(p)
    return sorted(out)


def write_json(obj, path):
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
