"""Tests for dataset tokenizer compatibility paths."""

import torch

from dimba.data.dataset import TextDataset


class DummyCallableTokenizer:
    def __call__(self, text, max_length, padding, truncation, return_tensors):
        assert isinstance(text, str)
        assert truncation is True
        assert return_tensors == "pt"
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        if padding == "max_length" and max_length > ids.shape[1]:
            pad = max_length - ids.shape[1]
            ids = torch.nn.functional.pad(ids, (0, pad), value=0)
        return {"input_ids": ids}


class DummyEncodeTokenizer:
    def encode(self, text):
        assert isinstance(text, str)
        return [7, 8, 9]


def test_text_dataset_with_callable_tokenizer():
    dataset = TextDataset(["hello world"], tokenizer=DummyCallableTokenizer(), max_length=5)
    sample = dataset[0]

    assert sample["input_ids"].tolist() == [1, 2, 3, 0, 0]


def test_text_dataset_with_encode_tokenizer():
    dataset = TextDataset(["hello world"], tokenizer=DummyEncodeTokenizer(), max_length=5)
    sample = dataset[0]

    assert sample["input_ids"].tolist() == [7, 8, 9, 0, 0]
