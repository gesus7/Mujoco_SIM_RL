#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v git >/dev/null
command -v nvidia-smi >/dev/null
python3 -m venv .venv-server
source .venv-server/bin/activate
mkdir -p reports
if [[ -f playground-0.2.0-py3-none-any.whl && -f panda-assets.zip ]]; then
    python scripts/prepare_offline.py
    python -m pip install -r reports/requirements-resolved.txt
else
    python -m pip install -r requirements-server.txt
    python scripts/fetch_assets.py
fi
python -m pip check
python -m pip freeze > reports/server-installed.txt
echo 'Installed. Activate with: source .venv-server/bin/activate'
echo 'Next: python -m panda_rl.doctor --gpu 0 --output reports/doctor-gpu0.json'
