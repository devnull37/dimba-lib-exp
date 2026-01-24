#!/usr/bin/env python3
"""Test script to verify DIMBA configuration and dependencies."""

import sys
import torch
import yaml

# Add src to path
sys.path.insert(0, str(__file__).rsplit('/', 1)[0] + '/../src')

def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")
    
    try:
        import pytorch_lightning as pl
        print("[OK] PyTorch Lightning available")
    except ImportError:
        print("[FAIL] PyTorch Lightning not available")
        return False
    
    try:
        from datasets import load_dataset
        print("[OK] HuggingFace datasets available")
    except ImportError:
        print("✗ HuggingFace datasets not available")
        return False
    
    try:
        from tokenizers import Tokenizer
        print("[OK] HuggingFace tokenizers available")
    except ImportError:
        print("✗ HuggingFace tokenizers not available")
        return False
    
    try:
        from mamba_ssm import Mamba2
        print("[OK] mamba-ssm available")
    except ImportError:
        print("[WARN] mamba-ssm not available (will use SimpleMamba2)")
    
    return True

def test_gpu():
    """Test GPU availability and memory."""
    print("\nTesting GPU...")
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Version: {torch.version.cuda}")
        return True
    else:
        print("[FAIL] CUDA not available")
        return False

def test_model_config():
    """Test if model configuration is reasonable."""
    print("\nTesting model configuration...")
    
    config = {
        'd_model': 2048,
        'd_prompt': 2048,
        'num_diffusion_steps': 1000,
        'num_denoiser_layers': 24,
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
        'conditioning_type': 'film',
        'dropout': 0.1,
        'use_weight_tying': True,
        'use_simple_mamba': False,
    }
    
    # Estimate parameter count
    d_model = config['d_model']
    d_state = config['d_state']
    expand = config['expand']
    num_layers = config['num_denoiser_layers']
    
    # Rough estimate for Mamba-2 parameters
    # Embedding: vocab_size * d_model
    # Mamba layers: num_layers * (d_model * d_model * expand + d_model * d_state + d_model * d_conv)
    # Output projection: d_model * vocab_size (if not weight-tied)
    
    vocab_size = 32000
    embedding_params = vocab_size * d_model
    mamba_params_per_layer = d_model * d_model * expand + d_model * d_state + d_model * d_conv
    mamba_params = num_layers * mamba_params_per_layer
    output_params = d_model * vocab_size if not config['use_weight_tying'] else 0
    
    total_params = embedding_params + mamba_params + output_params
    
    print(f"[OK] Model configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    print(f"  d_state: {d_state}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  Estimated parameters: {total_params:,}")
    print(f"  Model size (FP32): ~{total_params * 4 / 1e9:.1f} GB")
    print(f"  Model size (FP16): ~{total_params * 2 / 1e9:.1f} GB")
    
    # Check if it fits in 32GB GPU
    model_size_fp16 = total_params * 2 / 1e9
    batch_size = 16
    seq_length = 512
    activation_memory = batch_size * seq_length * d_model * 4 / 1e9  # FP32 activations
    
    total_memory = model_size_fp16 + activation_memory
    
    print(f"  Estimated memory usage:")
    print(f"    Model (FP16): {model_size_fp16:.1f} GB")
    print(f"    Activations: {activation_memory:.1f} GB")
    print(f"    Total: {total_memory:.1f} GB")
    
    if total_memory < 28:  # Leave some headroom
        print("[OK] Configuration should fit in 32GB GPU")
        return True
    else:
        print("[FAIL] Configuration may not fit in 32GB GPU")
        return False

def test_dataset_access():
    """Test if we can access the FineWeb dataset."""
    print("\nTesting dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to load a small sample
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            "sample-100BT",
            split="train",
            streaming=True
        )
        
        # Get first example
        example = next(iter(dataset))
        print("[OK] FineWeb dataset accessible")
        print(f"  Example keys: {list(example.keys())}")
        print(f"  Text length: {len(example.get('text', ''))}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed to access FineWeb dataset: {e}")
        return False

def main():
    print("=" * 60)
    print("DIMBA Configuration Test")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("GPU", test_gpu),
        ("Model Config", test_model_config),
        ("Dataset Access", test_dataset_access),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[FAIL] {name} test failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Ready to start training.")
    else:
        print("\n[WARNING] Some tests failed. Please fix issues before training.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)