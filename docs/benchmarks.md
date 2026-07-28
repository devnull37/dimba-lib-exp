# DIMBA Comparative Benchmark

Models benchmarked: DIMBA (masked discrete diffusion LM, LLaDA/MDLM objective, on a bidirectional Mamba backbone), SmolLM-135M, SmolLM-135M-Instruct (or SmolLM2-135M-Instruct as fallback), GPT-2 (124M), Pythia-160M. Parameter counts in Test E are measured directly (sum of p.numel()).

All AR models were tested with greedy decoding and sampled decoding (temp 0.7, top_k 20, seed 11). The better score per metric is reported, with the winning decode mode noted.

## Test A: Factual QA Keyword Accuracy

40 world-knowledge questions scored by keyword presence (case-insensitive) in the 40-token generated answer.

| Model | QA Accuracy | Decode Mode |
|-------|------------|-------------|
| DIMBA | 15.0% | diffusion (128 steps) |
| SmolLM-135M | 82.5% | greedy |
| HuggingFaceTB/SmolLM-135M-Instruct | 60.0% | greedy |
| GPT-2 (124M) | 20.0% | greedy |
| Pythia-160M | 10.0% | sampled(t=0.7,top_k=20) |

## Test B: Repetition and Degeneracy

Computed over the 40 QA answers per model. Distinct-1/2 = unique unigrams/bigrams over total (higher is better). Loop rate = fraction of answers with any 3-gram repeated 3+ times (lower is better).

| Model | Distinct-1 | Distinct-2 | Loop Rate |
|-------|-----------|-----------|----------|
| DIMBA | 0.288 | 0.676 | 7.5% |
| SmolLM-135M | 0.322 | 0.579 | 37.5% |
| HuggingFaceTB/SmolLM-135M-Instruct | 0.365 | 0.677 | 2.5% |
| GPT-2 (124M) | 0.140 | 0.258 | 90.0% |
| Pythia-160M | 0.347 | 0.700 | 15.0% |

## Test C: Infill Recovery

12 sentences from SENTS. Middle 50% of tokens masked. DIMBA fills natively (argmax at t=0.5, one shot). AR models receive the prefix and suffix with '____' and are prompted to fill the blank; token-level exact-match recovery of the masked span is measured. The asymmetry is expected: this is a diffusion-native capability.

| Model | Infill Recovery | Native Infill |
|-------|----------------|--------------|
| DIMBA | 14.0% | Yes |
| SmolLM-135M | 2.9% | No |
| HuggingFaceTB/SmolLM-135M-Instruct | 0.0% | No |
| GPT-2 (124M) | 0.0% | No |
| Pythia-160M | 1.7% | No |

## Test D: Latency

Median wall-clock seconds per 40-token answer, batch size 1, 1 warmup, 5 measured. DIMBA runs 128 diffusion steps per answer, which is reflected in its latency.

| Model | Latency (median) |
|-------|-----------------|
| DIMBA | 13.33s |
| SmolLM-135M | 0.63s |
| HuggingFaceTB/SmolLM-135M-Instruct | 0.62s |
| GPT-2 (124M) | 0.18s |
| Pythia-160M | 0.19s |

The published 13.33 s result predates CUDA graphs. A directional RTX 4090 verification on
2026-07-17 measured the current eager path at **17.92 s** and denoiser graph replay at
**1.16 s (15.49x)** for this production shape, with exact final tokens (40/40, seed 0).
This is a one-run 4090 measurement, not a replacement five-run H100 median; that formal rerun
remains tracked in `AGENTS.md`.

## Test E: Model Facts

| Model | Parameters | Native Infill |
|-------|-----------|--------------|
| DIMBA | 287.9M | Yes |
| SmolLM-135M | 134.5M | No |
| HuggingFaceTB/SmolLM-135M-Instruct | 134.5M | No |
| GPT-2 (124M) | 124.4M | No |
| Pythia-160M | 162.3M | No |

## Summary

DIMBA is capacity-bound and trained on far less data than the AR baselines, so it is expected to lose raw factual QA accuracy to SmolLM-135M, which benefits from standard autoregressive pre-training on a large corpus. Note that DIMBA's measured numel (288M) is larger than SmolLM's because the bidirectional Mamba backbone runs a state-space stack per direction and each of the 30 layers carries AdaLN timestep-conditioning parameters; the comparison class is still small models. The interesting axes are:

- Raw QA: SmolLM-135M wins decisively (82.5% vs DIMBA's 15.0%). No spin: on plain factual recall DIMBA is far behind its AR teacher.
- Native infill: DIMBA recovers 14.0% of masked middle spans in a single forward pass. Every AR baseline is at or near zero when prompted to fill the blank. This is a structural capability, not a tuning artifact.
- Degeneracy: DIMBA's loop rate is 7.5% versus 37.5% for SmolLM-135M; the diffusion sampler with frequency penalty degenerates less at this budget. The instruct-tuned baseline is the cleanest overall.
- Latency cost: DIMBA takes 13.33s per 40-token answer at 128 diffusion steps versus 0.63s for SmolLM. Cost scales with step count, so fewer steps trade quality for speed.
- Controllability: CFG (scale 2.0) and the frequency penalty give DIMBA levers that AR greedy decoding lacks by default.
