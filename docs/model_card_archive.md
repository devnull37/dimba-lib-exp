---
language: en
license: apache-2.0
tags:
  - diffusion-language-model
  - masked-diffusion
  - mamba
  - mdlm
  - llada
  - experimental
---

# d1-135m — bidirectional-Mamba masked-diffusion LM (135M)

An experimental **masked discrete diffusion language model** (LLaDA/MDLM-style objective) on a **bidirectional Mamba** backbone, 135M parameters, trained end-to-end for ~$450 total on rented H100s — including the failed continuous-diffusion first attempt whose checkpoints are also in this repo, because negative results are results. To our knowledge every published masked-diffusion text model is a transformer; the Mamba backbone is the experimental variable here.

**Headline demo** — same model, same prompt, classifier-free guidance off vs on:

> Q: What is the capital of France?
> **guidance 0:** " The name of the world is the world of the world of the world..."
> **guidance 2.0:** " **The European city of Paris is the city of of Paris.** It is in the city city of the Paris."

**Capacity demo** — the model memorizes facts but cannot generalize the schema:

> Q: What is the capital of Japan?
> " The capital of Paris is Paris, Paris, France, France..."

One drilled fact ("capital → Paris") fit in 135M parameters; "capital of X" did not.

## Files

| File | What it is |
|---|---|
| `mdm_sft_cfg2_final.pt` | ⭐ **Best model.** 422k-pair SFT (Alpaca + SmolTalk + math) with CFG dropout, 15k steps |
| `mdm_repair_final.pt` | cfg2 + repair fine-tune on random planted errors: error detection 0% → 7.1%, clean-token retention 99.2% |
| `critic_head.pt` | 300k-param token-critic MLP on the frozen cfg2 model: flags wrong tokens at 52.5% precision@k (chance 10%), 78.9% pairwise AUC (chance 50%) |
| `mdm_sft_cfg_final.pt` | First CFG SFT (Alpaca only) — the original "Paris" model |
| `mdm2_sft_final.pt` | SFT on the extended (fresh-data) base, no CFG |
| `mdm_sft2_final.pt` | SFT v2 on the first base |
| `mdm_sft_final.pt` | SFT v1 — **EOS-collapse failure**, kept for the record: loss trained on padding → model answers with empty strings while loss looks great |
| `mdm2_final.pt` | Base #2: +40k steps on 600k fresh docs. Negative result: identical metrics to base #1 → capacity-bound at 135M |
| `mdm_final.pt` | Base #1: 40k steps masked-diffusion conversion of the 28B-token backbone |
| `kd_final.pt` | Continuous-diffusion track KD attempt (negative result) |

## How to sample (current best recipe)

MaskGIT-style iterative unmasking:

- **128 unmasking steps** (measurably better than 64 — commit fewer tokens per step)
- temperature **0.7**, top-k **20**
- **CFG guidance 2.0**: `logits = uncond + 2.0 · (cond − uncond)` (uncond = prompt fully masked)
- **exempt-first frequency penalty 0.7**: penalize `count − 1` occurrences, so first use is free
- **neighbor ban**: a token may not be committed adjacent to an identical token

Tokenizer: `HuggingFaceTB/SmolLM-135M` + 1 `[MASK]` token (id 49152). Reference implementation: `scripts/` in the training repo.

## Training summary

1. **Backbone:** bidirectional Mamba, 135M, distilled from SmolLM-135M over 28B tokens (continuous-diffusion objective — failed for generation, but the English knowledge survived and powers everything below).
2. **Masked-diffusion conversion:** mask ratio t ~ U[0.03, 1], CE on masked positions, 1/t weighting; 40k steps, batch 64, seq 512, FineWeb sample-10BT.
3. **SFT:** 422k pairs (Alpaca 52k + SmolTalk 350k + math 20k), prompt never masked, loss on response + exactly one EOS (not the padding tail — see failure above), 15k steps. 10% of rows trained with fully-masked prompt for CFG.
4. **Repair fine-tune** (optional checkpoint): 15% of visible response tokens replaced with random tokens, loss on masked+corrupted positions → teaches planted-error detection. Training on the model's *own* sampled errors instead destroys this ability (echo chamber — see writeup).
5. **Critic head:** frozen base; MLP(576→256→1) on decoded-latent features; BCE on planted errors (50% random / 50% model-sampled hard negatives), 2k steps, ~$2.50.
6. **Optimizer note:** controlled 2k-step A/B of Muon vs AdamW on this architecture: Muon 5.453 vs AdamW 5.470 final CE, stable throughout — first Muon result on a Mamba diffusion LM we're aware of.

## Known limitations (honest)

- Repetition loops within sentences (capacity-bound: fresh data and longer SFT demonstrably did not help at this size).
- Facts are memorized, not generalized (see the Japan demo above).
- No arithmetic — math SFT taught equation *format*, not computation.
- Self-correction does not work at this scale: the model's confidence cannot distinguish wrong from specific, falling-confidence remasking never triggers (confidence in committed errors only grows), and perplexity-based reranking selects degenerate repetition. Use the external critic head instead.
- Research artifact, not a usable assistant.

## Author

Faris Allafi, 14, self-funded. Full writeup with the complete failure/fix chronology: *(link)*.
