"""Simple character-level tokenizer for CPU/testing."""

from typing import List, Union
import torch
import json
from .base import BaseTokenizer


class SimpleCharacterTokenizer(BaseTokenizer):
    """Simple character-level tokenizer.

    Maps characters to token IDs. Useful for testing and CPU training.
    """

    def __init__(self, vocab_size: int = 256):
        """Initialize character tokenizer.

        Args:
            vocab_size: Size of vocabulary (typically 256 for ASCII)
        """
        super().__init__(vocab_size)
        # Build character to token mapping
        self.char_to_id = {}
        self.id_to_char = {}

        # Reserve first few tokens for special tokens
        special_tokens = ['<pad>', '<unk>', '<eos>']
        for i, token in enumerate(special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token

        # Map ASCII characters
        next_id = len(special_tokens)
        for i in range(32, min(127, vocab_size - next_id + 32)):
            char = chr(i)
            self.char_to_id[char] = next_id
            self.id_to_char[next_id] = char
            next_id += 1

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs.

        Args:
            text: Text or list of texts

        Returns:
            Token IDs or list of token ID lists
        """
        if isinstance(text, str):
            return self._encode_single(text)
        else:
            return [self._encode_single(t) for t in text]

    def _encode_single(self, text: str) -> List[int]:
        """Encode single text string."""
        token_ids = []
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Use unknown token for unmapped characters
                token_ids.append(self.unk_token_id)
        return token_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chars = []
        for token_id in token_ids:
            if isinstance(token_id, (list, torch.Tensor)):
                # Handle nested lists
                token_id = int(token_id) if isinstance(token_id, torch.Tensor) else token_id[0]

            token_id = int(token_id)
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                # Skip special tokens in output
                if not char.startswith('<'):
                    chars.append(char)

        return ''.join(chars)

    def save(self, path: str) -> None:
        """Save tokenizer to disk.

        Args:
            path: Path to save tokenizer
        """
        data = {
            'vocab_size': self.vocab_size,
            'char_to_id': self.char_to_id,
            'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load tokenizer from disk.

        Args:
            path: Path to load tokenizer from
        """
        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab_size = data['vocab_size']
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
