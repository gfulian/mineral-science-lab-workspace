#!/usr/bin/env bash
set -euo pipefail

echo "Preparing Mineral Science laboratory environment..."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ -d "course/python" ]; then
    python -m pip install -e course/python
fi

# Git on Windows does not always preserve executable bits in ZIP-based
# workflows. Enforce them inside the Linux Codespace as a second safeguard.
chmod +x scripts/update-course scripts/start-lab

sudo ln -sf "$PWD/scripts/update-course" /usr/local/bin/update-course
sudo ln -sf "$PWD/scripts/start-lab" /usr/local/bin/start-lab

echo
echo "Installed course environment:"
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in (
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "gemmi",
    "spglib",
    "py3Dmol",
    "quantas",
):
    try:
        print(f"  {package:12s} {version(package)}")
    except PackageNotFoundError:
        print(f"  {package:12s} NOT INSTALLED")
PY

echo
echo "Environment ready."
echo "Use 'update-course' before starting newly released material."
echo "Use 'start-lab 01' to create a personal working copy of Laboratory 1."
