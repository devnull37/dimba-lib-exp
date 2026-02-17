#!/bin/bash
set -euo pipefail

echo "=== DIMBA setup for RTX A4000 (16GB) ==="

python -m pip install --upgrade pip

echo "Installing PyTorch (CUDA 12.1 wheels)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing DIMBA with GPU extras..."
pip install -e ".[gpu,tracking]"

echo "Installing hub upload utility..."
pip install huggingface_hub

echo "Verifying CUDA..."
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    print("CUDA version:", torch.version.cuda)
PY

echo "=== Setup complete ==="
echo "Train with: python scripts/train_fineweb_500m_a4000.py"
