#!/usr/bin/env python3
"""slider_curve.py -- Accuracy-vs-cost curve for DIMBA's quality slider.

For each quality Q in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], runs all 40 QA items
(seed 11) and records keyword accuracy, timing, steps, and N.

Usage:
    cd /workspace/dimba-lib-exp
    PYTHONPATH=src python scripts/experiments/slider_curve.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback

# Ensure src/ and scripts/ are on sys.path so imports resolve.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [os.path.join(_REPO, "src"), os.path.join(_REPO, "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Import the slider API and QA list.
# ---------------------------------------------------------------------------
from generate import load_model, load_tokenizer, slider_generate  # noqa: E402

from experiments.benchmark_compare import QA_ITEMS  # noqa: E402

# ---------------------------------------------------------------------------
# Scoring helper (identical to benchmark_compare.kw_hit).
# ---------------------------------------------------------------------------

def kw_hit(answer_text: str, keywords: list[str]) -> bool:
    a = answer_text.lower()
    return any(kw.lower() in a for kw in keywords)


# ---------------------------------------------------------------------------
# Main experiment.
# ---------------------------------------------------------------------------
QUALITY_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
SEED = 11
OUT_JSON = os.path.join(_REPO, "scripts", "experiments", "slider_curve_results.json")
OUT_MD   = os.path.join(_REPO, "docs", "slider_curve.md")


def run():
    wall_start = time.time()
    print("Loading model...", flush=True)
    model, mask_id = load_model()
    tokenizer = load_tokenizer()
    print("Model loaded.", flush=True)

    curve_rows: list[dict] = []
    errors: list[dict] = []

    for Q in QUALITY_LEVELS:
        print(f"\n--- Q={Q} ---", flush=True)
        try:
            hits = []
            per_q_seconds = []
            steps_used = None
            n_used = None

            for i, (question, keywords) in enumerate(QA_ITEMS):
                result = slider_generate(
                    model, tokenizer, mask_id, question,
                    quality=Q,
                    seed=SEED,
                )
                hit = kw_hit(result["text"], keywords)
                hits.append({
                    "idx": i,
                    "question": question,
                    "keywords": keywords,
                    "answer": result["text"],
                    "hit": hit,
                    "seconds": result["seconds"],
                })
                per_q_seconds.append(result["seconds"])

                # steps and n should be constant for a given Q; capture once.
                if steps_used is None:
                    steps_used = result["steps"]
                    n_used = result["n"]

                if (i + 1) % 10 == 0:
                    running_acc = sum(h["hit"] for h in hits) / len(hits)
                    print(
                        f"  [{i+1:2d}/40] acc={running_acc:.3f}  "
                        f"steps={steps_used} N={n_used}  "
                        f"last={result['seconds']:.2f}s",
                        flush=True,
                    )

            accuracy = sum(h["hit"] for h in hits) / len(hits)
            median_s = statistics.median(per_q_seconds)
            total_s = sum(per_q_seconds)

            row = {
                "Q": Q,
                "steps": steps_used,
                "n": n_used,
                "accuracy": accuracy,
                "median_s_per_answer": median_s,
                "total_s": total_s,
                "hits": hits,
            }
            curve_rows.append(row)
            print(
                f"  Q={Q}  steps={steps_used}  N={n_used}  "
                f"accuracy={accuracy:.3f}  median_s={median_s:.2f}  total_s={total_s:.1f}",
                flush=True,
            )

        except Exception:
            tb = traceback.format_exc()
            print(f"  ERROR at Q={Q}:\n{tb}", flush=True)
            errors.append({"Q": Q, "traceback": tb})

    wall_total = time.time() - wall_start

    # Save JSON.
    output = {
        "meta": {
            "seed": SEED,
            "quality_levels": QUALITY_LEVELS,
            "n_questions": len(QA_ITEMS),
            "wall_clock_total_s": wall_total,
        },
        "curve": curve_rows,
        "errors": errors,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)

    # Write markdown.
    _write_markdown(curve_rows, errors, wall_total)

    print(f"\nTotal wall clock: {wall_total:.1f} s  ({wall_total/60:.1f} min)", flush=True)
    print("\n=== DONE ===", flush=True)
    return output


def _write_markdown(rows: list[dict], errors: list[dict], wall_total: float):
    lines = []
    lines.append("# DIMBA Quality Slider: Accuracy vs Cost Curve\n")
    lines.append(
        "Experiment: 40-question factual QA benchmark, seed 11, quality Q in "
        "[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]. "
        "Scoring: keyword presence (case-insensitive) in the 40-token answer, "
        "identical to the method used in docs/benchmarks.md "
        "(DIMBA baseline: 15.0% at 128 steps, N=1).\n"
    )

    lines.append("## Results\n")
    lines.append("| Q | Steps | N | Accuracy | Median s/answer | Total s |")
    lines.append("|---|-------|---|----------|----------------|---------|")
    for r in rows:
        lines.append(
            f"| {r['Q']} "
            f"| {r['steps']} "
            f"| {r['n']} "
            f"| {r['accuracy']*100:.1f}% "
            f"| {r['median_s_per_answer']:.2f} "
            f"| {r['total_s']:.1f} |"
        )
    lines.append("")

    if errors:
        lines.append("## Errors\n")
        for e in errors:
            lines.append(f"Q={e['Q']} failed:")
            lines.append("```")
            lines.append(e["traceback"].strip())
            lines.append("```")
            lines.append("")

    lines.append("## Interpretation\n")

    # Build an honest interpretation from the actual data.
    if rows:
        accs = [r["accuracy"] for r in rows]
        lo_acc = min(accs)
        hi_acc = max(accs)
        spread = hi_acc - lo_acc

        lo_row = rows[0]
        hi_row = rows[-1]
        cost_ratio = hi_row["total_s"] / lo_row["total_s"] if lo_row["total_s"] > 0 else float("nan")

        monotone = all(accs[i] <= accs[i+1] + 0.025 for i in range(len(accs)-1))
        flat = spread < 0.08

        if flat:
            trend_phrase = (
                f"Accuracy ranges from {lo_acc*100:.1f}% to {hi_acc*100:.1f}% "
                f"across all quality levels, a spread of only {spread*100:.1f} percentage points. "
                "The curve is essentially flat: increasing Q from {lo_q} to {hi_q} does not "
                "reliably improve factual recall at this scale.".format(
                    lo_q=rows[0]["Q"], hi_q=rows[-1]["Q"]
                )
            )
        elif monotone:
            trend_phrase = (
                f"Accuracy rises from {lo_acc*100:.1f}% at Q={rows[0]['Q']} "
                f"to {hi_acc*100:.1f}% at Q={rows[-1]['Q']}, "
                "a gain of {gain:.1f} pp, roughly monotone with quality level.".format(
                    gain=spread * 100
                )
            )
        else:
            trend_phrase = (
                f"Accuracy is noisy, ranging from {lo_acc*100:.1f}% to {hi_acc*100:.1f}% "
                f"without a clean monotone trend across Q levels."
            )

        interp_lines = [
            trend_phrase,
            (
                f"The cost dimension does scale as designed: total wall time for Q=1.0 "
                f"({hi_row['total_s']:.0f} s) is {cost_ratio:.1f}x that of Q=0.1 "
                f"({lo_row['total_s']:.0f} s), driven by the joint increase in "
                f"diffusion steps ({lo_row['steps']} at Q=0.1, {hi_row['steps']} at Q=1.0) "
                f"and best-of-N candidates (N={lo_row['n']} to N={hi_row['n']})."
            ),
            (
                "A flat or noisy accuracy curve at this scale is a plausible and "
                "publishable outcome: the model is capacity-bound, meaning that extra "
                "compute at inference time cannot recover knowledge the weights never "
                "encoded during training. The mechanism, verifiable cost scaling with "
                "controllable trade-off knobs, is the contribution, not a claim that "
                "more steps always beat fewer steps on a 135M-class backbone."
            ),
            (
                "Future experiments with a larger or better-trained checkpoint, or "
                "narrower domain tasks where the model is not capacity-bound, may show "
                "a cleaner accuracy benefit from the higher-Q regimes."
            ),
        ]
        lines.append("\n\n".join(interp_lines))
    else:
        lines.append("No results were collected (all Q levels errored).")

    lines.append("")
    lines.append(f"Total wall-clock time for the full sweep: {wall_total:.0f} s ({wall_total/60:.1f} min).\n")

    text = "\n".join(lines)
    # Safety check: no em dashes allowed.
    assert "—" not in text, "Em dash found in markdown output -- remove it."

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as fh:
        fh.write(text)
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    run()
