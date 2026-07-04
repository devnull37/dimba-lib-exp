# Bidirectional weight sharing A/B: double stack vs shared vs shared + per-direction LoRA

Three-arm controlled comparison, from scratch, 2000 steps per arm, AdamW, identical seed and data order (FineWeb, masked-diffusion objective, ~65M tokens per arm). The only variable is how the denoiser gets its two directions.

| Arm | Scheme | Unique trainable params | Tail CE (last 200 steps) | Wall minutes |
|---|---|---|---|---|
| A | Full double stack (current architecture) | 287.9M | 6.697 | 18.5 |
| B | One stack shared by both directions | 225.5M | 6.968 | 18.2 |
| C | Shared stack + per-direction LoRA (rank 16, 2.9M adapter params) | 228.4M | 6.797 | 20.0 |

Full loss curves are in `scripts/experiments/bidir_ab_results.json`.

## Reading

- Pure weight sharing (B) saves 62.5M parameters and costs 0.271 nats at 2k steps. Only the directional Mamba mixers (62.5M per direction) can be shared: the rest of the denoiser (timestep conditioning, merge projections, norms) is already direction-agnostic, so the savings ceiling is lower than the "two full stacks" intuition suggests.
- Adding tiny per-direction LoRA adapters (C) recovers 63 percent of that gap (0.271 to 0.100 nats) for 2.9M parameters. C is the best parameters-per-nat arm: 21 percent fewer parameters than A at a 0.100 nat penalty, and its curve was still converging toward A at step 2000.
- Caveat: 2000 steps from scratch measures early learning efficiency, not converged capacity. The full-double arm may retain an advantage at convergence that this probe cannot see. The result justifies including shared+LoRA (with rank and placement sweeps) in the pilot phase of the next training run, not adopting it blind.

Provenance: the shared + per-direction LoRA scheme was proposed by Faris Allafi during the 2026-07-04 session and tested the same afternoon on the same GPU.
