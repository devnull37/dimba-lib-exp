#!/usr/bin/env python3
"""Upload DIMBA training artifacts to the Hugging Face Hub."""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload DIMBA artifacts to HF Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. username/dimba-500m-fineweb")
    parser.add_argument("--artifacts-dir", default="./checkpoints/fineweb_500m_a4000", help="Directory with checkpoints and tokenizer")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create private model repo")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Provide --token or set HF_TOKEN before uploading.")

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory not found: {artifacts_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(artifacts_dir),
        repo_type="model",
        token=token,
        commit_message="Upload DIMBA 500M A4000 FineWeb checkpoint and tokenizer",
    )

    print(f"Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
