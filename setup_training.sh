#!/bin/bash
# Minimal setup for DIMBA training on TensorDock

set -euo pipefail

echo "=== DIMBA Training Setup ==="

# Resolve Python binary
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "Error: Neither python3 nor python is installed or on PATH."
        exit 1
    fi
fi

# Use an isolated virtual environment so setup works on PEP 668 systems
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    VENV_PATH="$VIRTUAL_ENV"
    echo "Using active virtual environment: $VENV_PATH"
else
    VENV_PATH=".venv"
    if [[ ! -d "$VENV_PATH" ]]; then
        echo "Creating virtual environment at $VENV_PATH..."
        "$PYTHON_BIN" -m venv "$VENV_PATH"
    else
        echo "Using existing virtual environment at $VENV_PATH"
    fi
fi

PIP_BIN="$VENV_PATH/bin/pip"
PYTHON_VENV_BIN="$VENV_PATH/bin/python"

# Keep packaging tooling current for better wheel compatibility
"$PYTHON_VENV_BIN" -m pip install --upgrade pip setuptools wheel

# 1. Install core dependencies
echo "Installing core dependencies..."
"$PIP_BIN" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"$PIP_BIN" install pytorch-lightning transformers datasets tokenizers pyyaml tqdm

# 2. Install Mamba-2 for GPU
echo "Installing Mamba-2 dependencies..."
"$PIP_BIN" install mamba-ssm causal-conv1d

# 3. Test installation
echo "Testing installation..."
"$PYTHON_VENV_BIN" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 4. Quick model test
echo "Testing model creation..."
"$PYTHON_VENV_BIN" -c "
import sys
sys.path.insert(0, 'src')
from dimba import DIMBA
model = DIMBA(
    vocab_size=32000,
    d_model=2048,
    d_prompt=2048,
    num_diffusion_steps=1000,
    num_denoiser_layers=24,
    d_state=64,
    d_conv=4,
    expand=2,
    conditioning_type='film',
    dropout=0.1,
    use_weight_tying=True,
    use_simple_mamba=False,
)
total_params = sum(p.numel() for p in model.parameters())
print(f'Model created successfully!')
print(f'Total parameters: {total_params:,}')
print(f'Model size (FP16): {total_params * 2 / 1e9:.1f} GB')
"

echo "=== Setup Complete ==="
echo "To start training: $PYTHON_VENV_BIN scripts/train_fineweb_1b.py"
echo "Tip: activate the environment with: source $VENV_PATH/bin/activate"
