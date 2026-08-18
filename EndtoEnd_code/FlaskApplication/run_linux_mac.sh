#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3.11 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
