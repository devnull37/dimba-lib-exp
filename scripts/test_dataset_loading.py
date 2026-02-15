#!/usr/bin/env python3
"""Test if FineWeb 10BT can be loaded without streaming."""

from datasets import load_dataset


def test_fineweb_loading():
    """Test loading FineWeb dataset."""
    print("Testing FineWeb 10BT dataset loading...")

    try:
        # Try to load a small sample first
        print("Loading small sample to test...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            "sample-10BT",
            split="train",
            streaming=False,
            trust_remote_code=True,
        )

        print("Dataset loaded successfully!")
        print(f"Dataset type: {type(dataset)}")

        # Try to get length (this might fail for large datasets)
        try:
            length = len(dataset)
            print(f"Dataset length: {length:,} examples")
        except Exception as e:
            print(f"Cannot get length: {e}")

        # Try to get first example
        try:
            example = dataset[0]
            print(f"First example keys: {list(example.keys())}")
            if "text" in example:
                print(f"Text length: {len(example['text'])} chars")
        except Exception as e:
            print(f"Cannot access first example: {e}")

    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("\nRecommendation: Use streaming=True for 10BT dataset")
        print("Reason: 300GB dataset may not fit in memory/disk cache")


if __name__ == "__main__":
    test_fineweb_loading()
