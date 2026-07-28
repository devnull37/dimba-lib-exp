# CLAUDE.md

Read **AGENTS.md** first. It is the durable onboarding guide for this inference-only public
repository: scope and privacy boundary, current status, live TODOs, verification rules, and
invariants.

Quick pointers (details all in AGENTS.md):

- Run tests: `python3 -m pytest tests/ -q -p no:cacheprovider --override-ini addopts=`
- Release evidence: `docs/benchmarks.md`, `docs/slider_curve.md`, and
  `docs/hr-diffuse-1-nano_system_card.md`.
- Backend behavior and measured Mac results: `docs/BACKENDS.md`.
- Do not add training, distillation, transmutation, private scale-run, or cloud orchestration
  work to this public repository.
- Do not reintroduce: the prompt-conditioning leak, 2-tuple `forward`, un-scaled latents,
  hot-loop host syncs, or bespoke Mamba kernels without the promotion gate.
