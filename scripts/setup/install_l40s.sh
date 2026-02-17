#!/bin/bash
# Install dependencies for DIMBA training on L40S 48GB

echo "=== DIMBA Setup for L40S 48GB ==="

echo "Installing PyTorch with CUDA 12.1 (L40S compatible)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing core dependencies..."
pip install pytorch-lightning transformers datasets tokenizers pyyaml tqdm

echo "Installing Mamba-2 for GPU acceleration..."
pip install mamba-ssm causal-conv1d

echo "Installing monitoring tools..."
pip install tensorboard wandb

echo "Verifying GPU access..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'CUDA Version: {torch.version.cuda}')
"

echo "=== Setup Complete ==="
echo "To start training: python scripts/train_fineweb_1b.py"
echo "Configuration: configs/fineweb_1.5b_l40s.yaml"
echo ""
echo "Expected performance:"
echo "- Model: 1.5B parameters (~0.9GB FP16)"
echo "- Batch size: 32"
echo "- Sequence length: 1024"
echo "- VRAM usage: ~4GB (44GB headroom)"