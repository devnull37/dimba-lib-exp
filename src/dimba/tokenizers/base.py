"""Base tokenizer class."""

from abc import ABC, abstractmethod
from typing import List, Union
import torch


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    def __init__(self, vocab_size: int):
        """Initialize tokenizer.

        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size

    @abstractmethod
    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs.

        Args:
            text: Text or list of texts to encode

        Returns:
            Token IDs or list of token ID lists
        """
        pass

    @abstractmethod
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save tokenizer to disk.

        Args:
            path: Path to save tokenizer
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer from disk.

        Args:
            path: Path to load tokenizer from
        """
        pass

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.vocab_size - 1

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return 0

    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return 1
