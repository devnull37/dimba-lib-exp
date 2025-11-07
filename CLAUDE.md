# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is implementing **DIMBA** (Diffusion-based Mamba architecture), a non-autoregressive text generation model that combines:
- Cosine-scheduled diffusion process for parallel denoising
- Mamba-2 state-space model (SSM) as the denoiser backbone
- Conditioning via prompt embeddings and timestep embeddings

The goal is to achieve faster parallel text generation compared to autoregressive transformers while maintaining output quality.

## Architecture Summary

### Core Components (from Section 3.2 of the paper)

1. **Token Embeddings**: Input tokens mapped to continuous embedding space via learned embedding matrix E, producing X₀ ∈ ℝ^(L×d)

2. **Prompt Encoder**: Lightweight MLP or frozen encoder that processes token embeddings to produce conditioning vector C ∈ ℝ^(L×d_c)

3. **Cosine Noise Schedule**: Follows Nichol & Dhariwal (2021) with formula:
   - ᾱ(t) = cos²((t/T + s)/(1 + s) · π/2), s = 0.008
   - β_t = 1 - ᾱ(t)/ᾱ(t-1)
   - X_T = √ᾱ(T)X₀ + √(1 - ᾱ(T))ε

4. **Timestep Embedding**: Sinusoidal positional encoding processed through MLP yielding τ(t) ∈ ℝ^d

5. **Mamba-2 Denoiser**: N Mamba-2 blocks taking (X_t, C, τ(t)) as input with either additive or Feature-wise Linear Modulation (FiLM) conditioning

6. **Output Projection**: Linear layer mapping denoised embeddings to token logits (optionally weight-tied with embedding matrix)

### Data Flow

```
Input Prompt
    ↓
Token Embeddings (X₀)
    ↙              ↖
Prompt Encoder      Noise Injection (Cosine Schedule)
    ↓                    ↓
Conditioning (C)    Noisy Embeddings (X_T)
    ↓                    ↓
         Mamba-2 Denoiser (with Timestep Embedding τ)
                    ↓
            Denoised Embeddings (X₀_pred)
                    ↓
            Output Projection to Logits
                    ↓
                Output Tokens
```

## Training Procedure

**Objective**: Learn to reverse the diffusion process at arbitrary timesteps.

```
For each training batch:
  1. Sample random timestep: t ~ Uniform(1, T)
  2. Create noisy embeddings: X_t = √ᾱ(t)X₀ + √(1 - ᾱ(t))ε
  3. Encode prompt: C = PromptEncoder(X₀)
  4. Create timestep embedding: τ = MLP(t)
  5. Predict: X_pred = Denoiser(X_t, C, τ)
  6. Loss: L = ||X_pred - X₀||²
  7. Update parameters via backpropagation
```

## Inference Procedure

**Goal**: Generate text of length L_gen by iterative denoising from noise.

```
1. Compute prompt conditioning: C = PromptEncoder(X_prompt)
2. Initialize with noise: X_T ~ N(0, I) ∈ ℝ^(L_gen×d)
3. Iterative denoising loop (t = T down to 1):
     - τ = MLP(t)
     - X_{t-1} = Denoiser(X_t, C, τ)
4. Final projection: X₀ → linear layer → softmax → output tokens
```

The number of diffusion steps T controls the speed-quality trade-off: lower T = faster but potentially lower quality.

## Key Implementation Notes

### Hyperparameters to Consider
- **T**: Number of diffusion steps (controls inference speed/quality trade-off)
- **d**: Embedding dimension
- **d_c**: Conditioning dimension (may equal d)
- **N**: Number of Mamba-2 blocks in denoiser
- **s**: Noise schedule constant (0.008 per paper)

### Conditioning Mechanisms
- **Additive**: Simple concatenation with noise
- **FiLM (Feature-wise Linear Modulation)**: γ(C) * X_t + β(C), where γ and β are learned from C

### Important Design Choices Requiring Implementation Decisions
1. **Prompt Encoder**: Frozen or trainable? Use existing pretrained encoder or train from scratch?
2. **Weight Tying**: Should output projection share weights with embedding matrix?
3. **Mamba-2 Architecture**: How many layers? What hidden dimension?
4. **Sampling During Inference**: Pure denoising or DDIM-style acceleration?

## Paper References

- **Main sections**: See `paper/main.txt` for full details
- **Figure 1**: Shows complete end-to-end architecture
- **Section 3.2**: Detailed component descriptions
- **Section 3.3-3.4**: Training and inference procedures
- **Section 4.1**: Hypothesized advantages (latency, coherence, controllability, reasoning, extensibility)
- **Section 4.2**: Known challenges (training cost, discrete-continuous gap, conditioning robustness, hyperparameter sensitivity)

## Dependencies & Libraries

When implementing, likely dependencies include:
- PyTorch (for tensor operations)
- Mamba-2 implementation (from `mamba-2` package or custom implementation)
- Hugging Face Transformers (for tokenizers, embeddings, reference models)

## Testing Strategy

Based on Section 4.2 challenges:
- Test discrete-continuous mapping accuracy for rare tokens
- Validate conditioning mechanisms (FiLM vs additive) across diverse prompts
- Benchmark inference latency with varying T values
- Compare generation quality against autoregressive baselines
