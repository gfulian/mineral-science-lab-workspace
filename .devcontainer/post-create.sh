#!/usr/bin/env bash
set -euo pipefail

echo "Preparing Mineral Science laboratory environment..."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ -d "course/python" ]; then
    python -m pip install -e course/python
fi

sudo ln -sf "$PWD/scripts/update-course" /usr/local/bin/update-course
sudo ln -sf "$PWD/scripts/start-lab" /usr/local/bin/start-lab

echo
echo "Environment ready."
echo "Use 'update-course' before starting newly released material."
echo "Use 'start-lab 01' to create a personal working copy of Laboratory 1."
