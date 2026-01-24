#!/bin/bash
# Minimal setup for DIMBA training on TensorDock

echo "=== DIMBA Training Setup ==="

# 1. Install core dependencies
echo "Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning transformers datasets tokenizers pyyaml tqdm

# 2. Install Mamba-2 for GPU
echo "Installing Mamba-2 dependencies..."
pip install mamba-ssm causal-conv1d

# 3. Test installation
echo "Testing installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 4. Quick model test
echo "Testing model creation..."
python -c "
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
echo "To start training: python scripts/train_fineweb_1b.py"