"""
Quick Start Guide for DIMBA

This script demonstrates basic usage of the DIMBA library:
1. Creating a model
2. Training on dummy data
3. Generating text
4. Evaluating the model

To run this as a notebook, convert with:
    jupyter nbconvert --to notebook quickstart.py
"""

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

import torch
from torch.utils.data import DataLoader

from dimba import DIMBA, sample_from_model, DDIMSampler
from dimba.data import DummyDataset, collate_fn
from dimba.training import DIMBALightningModule
from dimba.evaluation import compute_model_perplexity, MetricsLogger
import pytorch_lightning as pl


print("=" * 70)
print("DIMBA Quick Start Guide")
print("=" * 70)

# ============================================================================
# 1. CREATE MODEL
# ============================================================================

print("\n1. Creating DIMBA Model")
print("-" * 70)

model = DIMBA(
    vocab_size=10000,        # Small vocab for quick testing
    d_model=256,             # Smaller dimension for faster training
    d_prompt=256,
    num_diffusion_steps=100, # Fewer steps for quick testing
    num_denoiser_layers=4,   # Fewer layers
    d_state=8,
    conditioning_type="film",
)

print(f"Model created successfully!")
print(f"  Vocabulary size: {model.vocab_size}")
print(f"  Model dimension: {model.d_model}")
print(f"  Number of diffusion steps: {model.num_diffusion_steps}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================

print("\n2. Preparing Data")
print("-" * 70)

# Create dummy dataset for demonstration
train_dataset = DummyDataset(size=100, vocab_size=10000, seq_length=64)
val_dataset = DummyDataset(size=20, vocab_size=10000, seq_length=64)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
)

print(f"Training set: {len(train_dataset)} examples")
print(f"Validation set: {len(val_dataset)} examples")
print(f"Batch size: 8")
print(f"Sequence length: 64")

# ============================================================================
# 3. TRAINING SETUP
# ============================================================================

print("\n3. Setting Up Training")
print("-" * 70)

# Create Lightning module
lightning_module = DIMBALightningModule(
    vocab_size=10000,
    model_config={
        'd_model': 256,
        'd_prompt': 256,
        'num_diffusion_steps': 100,
        'num_denoiser_layers': 4,
    },
    learning_rate=1e-4,
    warmup_steps=50,
    use_ema=True,
)

print("Lightning module created")
print(f"  Learning rate: 1e-4")
print(f"  Warmup steps: 50")
print(f"  EMA enabled: Yes")

# ============================================================================
# 4. QUICK INFERENCE TEST (before training)
# ============================================================================

print("\n4. Testing Forward Pass (Inference)")
print("-" * 70)

model.eval()
with torch.no_grad():
    # Encode prompt
    prompt_ids = torch.randint(0, 10000, (1, 8))
    prompt_cond = model.encode_prompt(prompt_ids)
    print(f"Prompt encoding shape: {prompt_cond.shape}")

    # Generate embeddings
    seq_len = 32
    x_t = torch.randn(1, seq_len, model.d_model)
    print(f"Initial noise shape: {x_t.shape}")

    # Single denoising step
    t = torch.tensor([50])
    x_pred = model.denoise_step(x_t, t, prompt_cond[:, :seq_len, :])
    print(f"Denoised embeddings shape: {x_pred.shape}")

    # Project to logits
    logits = model.output_head(x_pred)
    print(f"Output logits shape: {logits.shape}")

# ============================================================================
# 5. QUICK TRAINING (optional - commented out for speed)
# ============================================================================

print("\n5. Training Setup (Can be run with trainer.fit)")
print("-" * 70)
print("To train the model, uncomment and run:")
print("""
trainer = pl.Trainer(
    max_epochs=1,
    accelerator='auto',
    devices=1,
    log_every_n_steps=10,
)
trainer.fit(lightning_module, train_loader, val_loader)
""")

# Uncomment to actually train:
# trainer = pl.Trainer(
#     max_epochs=1,
#     accelerator='auto',
#     devices=1 if torch.cuda.is_available() else None,
#     log_every_n_steps=10,
# )
# trainer.fit(lightning_module, train_loader, val_loader)

# ============================================================================
# 6. SAMPLING / GENERATION
# ============================================================================

print("\n6. Text Generation (Sampling)")
print("-" * 70)

model.eval()
prompt_ids = torch.randint(0, 10000, (1, 10))

print("Using standard denoising sampling...")
generated = sample_from_model(
    model,
    prompt_ids,
    seq_len=20,
    num_steps=10,  # Use few steps for quick demo
    temperature=1.0,
    top_k=None,
    top_p=0.95,
)

print(f"Generated shape: {generated.shape}")
print(f"Generated IDs (first 20): {generated[0, :20].tolist()}")

# ============================================================================
# 7. DDIM SAMPLING
# ============================================================================

print("\n7. Accelerated Sampling (DDIM)")
print("-" * 70)

sampler = DDIMSampler(model, num_steps=5, ddim_eta=0.0)
print("Using DDIM sampler with 5 steps...")

generated_ddim = sampler.sample(
    prompt_ids,
    seq_len=20,
    temperature=1.0,
)

print(f"Generated shape: {generated_ddim.shape}")
print(f"Generated IDs (first 20): {generated_ddim[0, :20].tolist()}")

# ============================================================================
# 8. EVALUATION
# ============================================================================

print("\n8. Model Evaluation")
print("-" * 70)

model.eval()
ppl = compute_model_perplexity(model, val_loader, device='cpu')
print(f"Model Perplexity on validation set: {ppl:.4f}")

# ============================================================================
# 9. ACCESSING COMPONENTS
# ============================================================================

print("\n9. Accessing Model Components")
print("-" * 70)

# Get embeddings
alphas = model.get_alphas_cumprod()
print(f"Noise schedule alphas shape: {alphas.shape}")
print(f"Alphas range: [{alphas.min():.4f}, {alphas.max():.4f}]")

# Get inference model (EMA if available)
from dimba.training import DIMBALightningModule
lightning_module = DIMBALightningModule(
    vocab_size=10000,
    model_config={'d_model': 256, 'num_denoiser_layers': 4}
)
infer_model = lightning_module.get_model_for_inference()
print(f"Inference model (EMA): {type(infer_model).__name__}")

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("Quick Start Complete!")
print("=" * 70)

print("""
Next Steps:
1. Train a model: python scripts/train.py --config config.yaml
2. Generate text: python scripts/generate.py --checkpoint <path>
3. Evaluate: python scripts/evaluate.py --checkpoint <path>
4. Read the paper: paper/main.txt
5. Check CLAUDE.md for architecture details

Key Features to Explore:
- Different conditioning types: 'film' vs 'additive'
- Number of diffusion steps: affects speed/quality trade-off
- Model dimensions: try smaller for testing, larger for better quality
- Sampling strategies: standard denoising vs DDIM
- Datasets: dummy, HuggingFace, or custom text

For more details, see:
- README.md: Installation, configuration, and usage
- CLAUDE.md: Architecture and development guide
- paper/main.txt: Full paper with technical details
""")

print("=" * 70)
