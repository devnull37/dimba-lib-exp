"""Dataset classes for DIMBA training."""

import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
from datasets import load_dataset


class TextDataset(Dataset):
    """Generic text dataset for DIMBA training.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer function or HuggingFace tokenizer
        max_length: Maximum sequence length
        padding: Whether to pad sequences
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 256,
        padding: bool = True,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if callable(self.tokenizer):
            # HuggingFace-style tokenizer
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length" if self.padding else "do_not_pad",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
        elif hasattr(self.tokenizer, "encode"):
            # Custom tokenizer class with encode API
            tokens = self.tokenizer.encode(text)
            input_ids = torch.tensor(tokens, dtype=torch.long)

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]

            if self.padding and len(input_ids) < self.max_length:
                pad_size = self.max_length - len(input_ids)
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_size), value=0)
        else:
            raise TypeError("Tokenizer must be callable or provide an encode(text) method")

        return {
            "input_ids": input_ids,
        }


class HuggingFaceDataset(Dataset):
    """Load datasets from HuggingFace datasets library.

    Args:
        dataset_name: Name of dataset from HuggingFace (e.g., "wikitext", "openwebtext")
        split: Dataset split to use (e.g., "train", "validation")
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_examples: Maximum number of examples to load (None for all)
        text_column: Name of text column in dataset
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        split: str = "train",
        tokenizer=None,
        max_length: int = 256,
        num_examples: Optional[int] = None,
        text_column: str = "text",
        streaming: bool = False,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.streaming = streaming

        # Load dataset
        try:
            self.dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                split=split,
                streaming=streaming,
            )
        except Exception as e:
            raise ValueError(f"Failed to load dataset {dataset_name}: {e}")

        # Limit examples if not streaming
        if num_examples is not None and not streaming:
            self.dataset = self.dataset.select(range(min(num_examples, len(self.dataset))))
        elif num_examples is not None and streaming:
            self.dataset = self.dataset.take(num_examples)

    def __len__(self):
        if self.streaming:
            # For streaming datasets, return a large number or use num_examples if provided
            # This is a workaround since streaming datasets don't have a defined length
            return 1000000  # Return a large number for streaming mode
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        text = example[self.text_column]

        if self.tokenizer is not None:
            if callable(self.tokenizer):
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].squeeze(0)
            elif hasattr(self.tokenizer, "encode"):
                tokens = self.tokenizer.encode(text)
                input_ids = torch.tensor(tokens, dtype=torch.long)

                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]

                if len(input_ids) < self.max_length:
                    pad_size = self.max_length - len(input_ids)
                    input_ids = torch.nn.functional.pad(input_ids, (0, pad_size), value=0)
            else:
                raise TypeError("Tokenizer must be callable or provide an encode(text) method")
        else:
            # Simple word-based tokenization
            tokens = text.split()[:self.max_length]
            input_ids = torch.tensor(tokens, dtype=torch.long)

            # Pad
            if len(input_ids) < self.max_length:
                pad_size = self.max_length - len(input_ids)
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_size), value=0)

        return {
            "input_ids": input_ids,
        }


class DummyDataset(Dataset):
    """Dummy dataset for testing and debugging.

    Generates random token sequences.

    Args:
        size: Number of examples
        vocab_size: Size of vocabulary
        seq_length: Sequence length
    """

    def __init__(self, size: int = 100, vocab_size: int = 50000, seq_length: int = 256):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.long)
        return {"input_ids": input_ids}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for data loader.

    Args:
        batch: List of examples from dataset

    Returns:
        Batched tensors
    """
    input_ids = torch.stack([example["input_ids"] for example in batch])

    return {
        "input_ids": input_ids,
    }
