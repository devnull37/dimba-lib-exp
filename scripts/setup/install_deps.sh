#!/bin/bash
# Install dependencies for DIMBA training

echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing core dependencies..."
pip install pytorch-lightning transformers datasets tokenizers pyyaml tqdm

echo "Installing Mamba-2 for GPU acceleration..."
pip install mamba-ssm causal-conv1d

echo "Installing optional dependencies..."
pip install tensorboard

echo "Installation complete!"
echo "To verify GPU access: python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo "To test dataset: python -c \"from datasets import load_dataset; ds = load_dataset('HuggingFaceFW/fineweb', 'sample-10BT', streaming=True); print('Dataset accessible:', next(iter(ds)))\""