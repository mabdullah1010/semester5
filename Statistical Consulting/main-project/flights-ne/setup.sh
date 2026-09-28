#!/usr/bin/env bash
# One-time setup: venv, packages, download 2025 BTS data, build Northeast parquet.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
python scripts/download.py --year 2025
python scripts/build.py --year 2025
echo "Done. Activate with: source .venv/bin/activate"
