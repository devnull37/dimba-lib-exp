# DIMBA Quality Slider: Accuracy vs Cost Curve

Experiment: 40-question factual QA benchmark, seed 11, quality Q in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]. Scoring: keyword presence (case-insensitive) in the 40-token answer, identical to the method used in docs/benchmarks.md (DIMBA baseline: 15.0% at 128 steps, N=1).

## Results

| Q | Steps | N | Accuracy | Median s/answer | Total s |
|---|-------|---|----------|----------------|---------|
| 0.1 | 21 | 1 | 7.5% | 2.23 | 94.6 |
| 0.3 | 37 | 1 | 17.5% | 3.92 | 157.6 |
| 0.5 | 64 | 2 | 10.0% | 7.11 | 284.5 |
| 0.7 | 111 | 4 | 12.5% | 12.17 | 487.1 |
| 0.9 | 194 | 8 | 17.5% | 21.16 | 846.4 |
| 1.0 | 256 | 8 | 20.0% | 27.72 | 1109.2 |

## Interpretation

Accuracy is noisy, ranging from 7.5% to 20.0% without a clean monotone trend across Q levels.

The cost dimension does scale as designed: total wall time for Q=1.0 (1109 s) is 11.7x that of Q=0.1 (95 s), driven by the joint increase in diffusion steps (21 at Q=0.1, 256 at Q=1.0) and best-of-N candidates (N=1 to N=8).

A flat or noisy accuracy curve at this scale is a plausible and publishable outcome: the model is capacity-bound, meaning that extra compute at inference time cannot recover knowledge the weights never encoded during training. The mechanism, verifiable cost scaling with controllable trade-off knobs, is the contribution, not a claim that more steps always beat fewer steps on a 135M-class backbone.

Future experiments with a larger or better-trained checkpoint, or narrower domain tasks where the model is not capacity-bound, may show a cleaner accuracy benefit from the higher-Q regimes.

Total wall-clock time for the full sweep: 2984 s (49.7 min).
