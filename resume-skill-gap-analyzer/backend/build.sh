#!/usr/bin/env bash
# Render build script for the backend
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Build-time dataset processing: download HF data & generate training CSV
echo "Running build-time dataset processor..."
python build_datasets.py || echo "Dataset build failed (non-critical) — app will use synthetic fallback"
