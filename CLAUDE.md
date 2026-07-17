# CLAUDE.md

Please go read **AGENTS.md** for info — it is the single onboarding guide for this repo:
project overview, full history, current status, live TODO list, GPU verification protocol,
environment gotchas, and the invariants you must not break.

Quick pointers (details all in AGENTS.md):

- Run tests: `python3 -m pytest tests/ -q -p no:cacheprovider --override-ini addopts=`
  (543 green baseline).
- Perf truth (measured vs estimated): `docs/PERFORMANCE_AND_SCALING.md`.
- Next training run contract: `docs/NEXT_RUN_PLAN.md` (authoritative).
- CUDA benchmark + promotion gate: `scripts/benchmark_h100.py --preset production`.
- Do not reintroduce: the prompt-conditioning leak, 2-tuple `forward`, un-scaled latents,
  hot-loop host syncs, or bespoke Mamba kernels without the promotion gate.
