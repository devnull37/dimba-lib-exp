"""BPE tokenizer using HuggingFace tokenizers library."""

from typing import List, Union, Optional
import torch
from .base import BaseTokenizer

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


class BPETokenizer(BaseTokenizer):
    """BPE (Byte Pair Encoding) tokenizer using HuggingFace tokenizers.

    Provides efficient subword tokenization. Requires 'tokenizers' library.
    """

    def __init__(self, vocab_size: int = 10000, tokenizer_obj: Optional[Tokenizer] = None):
        """Initialize BPE tokenizer.

        Args:
            vocab_size: Size of vocabulary
            tokenizer_obj: Optional pre-initialized tokenizer object
        """
        if not HAS_TOKENIZERS:
            raise ImportError(
                "BPETokenizer requires 'tokenizers' library. "
                "Install with: pip install tokenizers"
            )

        super().__init__(vocab_size)

        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        else:
            # Create new BPE tokenizer
            self.tokenizer = Tokenizer(BPE())
            self.tokenizer.pre_tokenizer = Whitespace()

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 10000,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """Train BPE tokenizer on texts.

        Args:
            texts: List of texts to train on
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens (default: pad, unk, eos)

        Returns:
            Trained BPETokenizer instance
        """
        if not HAS_TOKENIZERS:
            raise ImportError(
                "BPETokenizer requires 'tokenizers' library. "
                "Install with: pip install tokenizers"
            )

        if special_tokens is None:
            special_tokens = ['<pad>', '<unk>', '<eos>']

        # Create trainer
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

        # Create tokenizer
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        # Train on texts
        tokenizer.train_from_iterator(texts, trainer=trainer)

        return cls(vocab_size=vocab_size, tokenizer_obj=tokenizer)

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs.

        Args:
            text: Text or list of texts

        Returns:
            Token IDs or list of token ID lists
        """
        if isinstance(text, str):
            encoded = self.tokenizer.encode(text)
            return encoded.ids
        else:
            return [self.tokenizer.encode(t).ids for t in text]

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Handle nested lists/tensors
        if token_ids and isinstance(token_ids[0], (list, torch.Tensor)):
            token_ids = token_ids[0]

        token_ids = [int(t) for t in token_ids]
        return self.tokenizer.decode(token_ids)

    def save(self, path: str) -> None:
        """Save tokenizer to disk.

        Args:
            path: Path to save tokenizer
        """
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        """Load tokenizer from disk.

        Args:
            path: Path to load tokenizer from
        """
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.tokenizer.token_to_id('<pad>')

    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.tokenizer.token_to_id('<unk>')

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.tokenizer.token_to_id('<eos>')
