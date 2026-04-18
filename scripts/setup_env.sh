#!/bin/bash
set -e
echo "Setting up environment..."
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
echo "Setup complete. Activate with: source venv/bin/activate"
