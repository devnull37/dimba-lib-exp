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

# hr-diffuse-1-nano

**A masked discrete diffusion language model on a bidirectional Mamba backbone. 287.9M measured parameters, 135M-class backbone capacity.**

This is the final release of the first generation of the project. It is a research artifact, trained end to end for about $450 on rented H100s by a self-funded independent researcher. To our knowledge, every published masked-diffusion text model uses a transformer backbone (LLaDA, MDLM, Dream). This model is the same proven objective on a different spine: bidirectional Mamba.

The full development archive, including every failed checkpoint and negative result, lives at [devnull37/d1-135m-28b](https://huggingface.co/devnull37/d1-135m-28b). This repo contains only the release artifacts.

## System card

### What it is

- **Architecture:** bidirectional Mamba with a timestep-conditioned denoiser and a token head. Measured parameter count: 287.9M (denoiser 258.1M, embeddings 28.3M, conditioning 1.6M). We label it 135M-class because the two directional stacks store largely redundant knowledge and the backbone was distilled from SmolLM-135M at the same hidden size; its measured failure profile matches that capacity. All parameters are active every forward pass. Tokenizer: `HuggingFaceTB/SmolLM-135M` plus one `[MASK]` token (id 49152).
- **Objective:** LLaDA/MDLM-style masked diffusion. Corrupt text by replacing a random fraction t of tokens with `[MASK]`, predict the originals with cross-entropy weighted 1/t.
- **Generation:** MaskGIT-style iterative unmasking. Start fully masked, repeatedly commit the most confident tokens.
- **Knowledge source:** a backbone distilled from SmolLM-135M over 28B tokens, converted to masked diffusion in 40k steps, then instruction-tuned on 422k pairs (Alpaca + SmolTalk + math) with classifier-free guidance dropout.

### What it can do

- Answer simple factual and topical prompts in grammatical English when sampled with the full recipe below. Example, real output:

  > Q: What is the capital of France?
  > "The European city of Paris is the city of of Paris. It is in the city city of the Paris."

- Stay on topic reliably: ocean questions get ocean words, France questions get country words.
- Trade cost for quality at inference time along two independent axes: number of denoising steps, and best-of-N with the included verifier head. This dial is native to diffusion and does not exist in autoregressive models of this size.

### What it cannot do (measured, not guessed)

- **Generalize facts.** It memorizes single associations. Asked for the capital of Japan, it answers Paris. One drilled fact fit in the weights; the schema "capital of X" did not.
- **Arithmetic.** Math SFT taught equation format, not computation.
- **Avoid repetition without help.** The raw model loops. The sampler recipe below is required.
- **Judge itself.** This is the central finding of the project. Six independent experiments (perplexity reranking, confidence-based remasking with two refill strategies, self-generated repair training, falling-confidence remasking, lookahead verification) all failed the same way: at 135M the model's confidence cannot distinguish wrong from specific, and its confidence in committed errors only grows as context fills in. Every working quality technique is an external constraint: classifier-free guidance, the anti-repeat sampler, and the critic head below.

### Is the model plus the critic head better than the model alone?

Honest answer: **better at detection, not yet better at generation.**

- The critic head finds planted wrong tokens with 52.5% precision at k (chance is 10%) and ranks a wrong token above a correct one 78.9% of the time (chance is 50%). The model's own confidence performs at roughly chance on the same tests. So as an error detector, model plus critic clearly beats the model alone.
- Wired into best-of-8 ranking, the critic reliably puts the most degenerate candidates last, but at 135M all candidates are low quality, so end accuracy barely moves.
- Wired into decode-time correction (critic picks tokens to remask, model refills), the critic score improves on every prompt but keyword accuracy does not, because the refill is still done by the same 135M model, which substitutes safe filler. A textbook Goodhart effect.

Conclusion: detection is solved externally at this scale; correction additionally needs a stronger generator. The pair becomes genuinely useful when the base model scales.

### Measured comparison against same-class models

Benchmarked 2026-07-04 on a 40-item factual QA set, a 12-sentence middle-50% infill test, and degeneracy metrics over the QA generations (full harness and raw results in the training repo, `docs/benchmarks.md`).

| Model | QA accuracy | Loop rate | Infill recovery | Seconds per 40-token answer |
|---|---|---|---|---|
| hr-diffuse-1-nano (this model) | 15.0% | 7.5% | 14.0% | 13.3 |
| SmolLM-135M (the AR teacher) | 82.5% | 37.5% | 2.9% | 0.63 |
| SmolLM-135M-Instruct | 60.0% | 2.5% | 0.0% | 0.62 |
| GPT-2 (124M) | 20.0% | 90.0% | 0.0% | 0.18 |
| Pythia-160M | 10.0% | 15.0% | 1.7% | 0.19 |

Read honestly: the model loses raw QA hard to its own teacher (capacity plus a lossy transfer pipeline), roughly matches GPT-2, and beats Pythia-160M despite an order of magnitude less training data. Its structural wins are native infill (every autoregressive baseline scores near zero because the task requires conditioning on both sides of a gap) and loop resistance (7.5% degenerate answers vs 37.5% for its teacher). Latency is its worst axis: 128 denoising steps with classifier-free guidance is 21x slower than the teacher.

An inference-time quality dial (`scripts/generate.py` in the training repo) maps one knob to denoising steps and best-of-N with a verifier: measured 2.2 to 27.7 seconds per answer across the dial, QA moving 7.5% to 20% (noisy and not monotone: the cost axis works perfectly, the accuracy axis is capacity-bound). Practical summary: 15% at production settings, 20% with the dial maxed, about 18% at settings you would actually wait for.

### Architecture note: cheaper bidirectionality

A controlled 3-arm A/B (2000 steps from scratch, identical data and seeds) tested whether the duplicated directional stacks can be shared: full double stack (287.9M, tail CE 6.697) vs one shared stack (225.5M, 6.968) vs shared stack plus per-direction LoRA rank 16 (228.4M, 6.797). The 2.9M LoRA adapters recover 63% of the quality lost to sharing while keeping a 21% parameter cut, making shared+LoRA the best parameters-per-nat variant tested. This is an early-learning probe, not a convergence result; the next run's pilot phase will confirm before adoption. Details: `docs/bidir_ab.md` in the training repo.

## Files

| File | What it is |
|---|---|
| `hr_diffuse_1_nano.pt` | The release model. 422k-pair SFT with CFG dropout, 15k steps |
| `hr_diffuse_1_nano_repair.pt` | Repair variant: fine-tuned on random planted errors. Error detection 0% to 7.1%, clean-token retention 99.2% |
| `critic_head.pt` | 300k-parameter token critic MLP over the frozen release model's decoded-latent features. Precision at k 52.5%, pairwise AUC 78.9% |

## How to sample (required recipe)

- 128 unmasking steps (measurably better than 64: fewer tokens committed per step)
- temperature 0.7, top-k 20
- classifier-free guidance 2.0: `logits = uncond + 2.0 * (cond - uncond)`, where uncond masks the whole prompt
- exempt-first frequency penalty 0.7: penalize `count - 1` occurrences, so first use is free
- neighbor ban: a token may not be committed adjacent to an identical token

## Training notes

1. SFT loss is computed on the response plus exactly one EOS token, never on the padding tail. Training on the tail silently collapses the model to empty answers while the loss looks excellent.
2. Repair training works with random corruptions and fails with self-generated ones. Training the model to fix its own sampled errors destroyed its detection ability entirely (7.1% to 0.0%), because its own samples are by definition what it finds plausible.
3. Optimizer: a controlled 2k-step A/B of Muon vs AdamW on this architecture gave Muon a small consistent win (final CE 5.453 vs 5.470) with no instability. We believe this is the first Muon result on a Mamba diffusion LM. The next scale-up trains with Muon.

## Roadmap

- Next run (when funded, roughly $1,500 to $4,000): 1.5B with teacher-enabled distillation from SmolLM2-1.7B, Muon optimizer, and the shared-base plus per-direction LoRA bidirectionality validated above. The run starts with a cheap pilot A/B phase (teacher on vs off, architecture ladder) before committing the budget.
- The scientific goal is a scaling curve for the four probes established here: planted-error detection rate, remask fire rate, critic AUC, and the slope of the inference-compute dial. All four are measured at this scale and waiting for their second data point.
- Planning-latent tokens: a compressed continuous plan vector conditioning the discrete diffusion.

## Author

Faris Allafi, 14, self-funded. Feedback from researchers is genuinely welcome.
