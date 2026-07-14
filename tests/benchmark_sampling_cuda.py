"""Manual eager-CUDA sampler benchmark.

Run on the target GPU, for example:
    python tests/benchmark_sampling_cuda.py --steps 20 --repeats 5

The output identifies the actual CUDA device; it does not extrapolate from MPS/CPU.
"""

import argparse
import json
import os
import statistics
import sys

import torch

from dimba.diffusion.sampling import sample_from_model
from dimba.models.diffusion import DIMBA


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required; run this helper on the target H100 host", file=sys.stderr)
        return 1

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = DIMBA(
        vocab_size=32_000,
        d_model=512,
        d_prompt=512,
        num_diffusion_steps=1_000,
        num_denoiser_layers=6,
        latent_diffusion=True,
        d_latent=256,
        dropout=0.0,
    ).to(device=device, dtype=torch.bfloat16)
    prompt = torch.randint(0, 32_000, (args.batch_size, 32), device=device)

    def run() -> None:
        sample_from_model(
            model,
            prompt,
            args.seq_len,
            num_steps=args.steps,
            guidance_scale=2.0,
            temperature=1.0,
        )

    for _ in range(2):
        run()
    torch.cuda.synchronize()

    timings = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "mode": "eager",
                "dtype": "bfloat16",
                "batch_size": args.batch_size,
                "prompt_len": 32,
                "seq_len": args.seq_len,
                "steps": args.steps,
                "guidance_scale": 2.0,
                "median_ms": statistics.median(timings),
                "samples_per_second": 1_000 * args.batch_size / statistics.median(timings),
                "runs_ms": timings,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
