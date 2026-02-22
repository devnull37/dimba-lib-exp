"""Unit tests for finetuning dataset utilities."""

import sys
import types

import torch

# ``dimba.data`` imports ``datasets`` at package import time; provide a tiny stub
# so these unit tests stay lightweight and offline-friendly.
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

from dimba.data.finetuning import (
    ChatDataset,
    ConcatenatedSequenceDataset,
    InstructionDataset,
    PreferenceDataset,
    resolve_dataset_source,
)


class TinyTokenizer:
    """Simple deterministic tokenizer for unit tests."""

    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if not text:
            return []
        # Keep token ids away from pad/eos for clearer assertions.
        return [3 + (ord(ch) % 50) for ch in text]


def _assert_standard_item(item, max_length):
    assert set(item) == {"input_ids", "attention_mask", "labels", "loss_mask"}
    for key in item:
        assert item[key].shape == (max_length,)
        assert item[key].dtype == torch.long


def test_instruction_dataset_formats_instruction_input_output() -> None:
    tokenizer = TinyTokenizer()
    data = [
        {
            "instruction": "Translate to Spanish",
            "input": "hello",
            "output": "hola",
        }
    ]
    dataset = InstructionDataset(data, tokenizer=tokenizer, max_length=128)

    item = dataset[0]
    _assert_standard_item(item, max_length=128)

    normalized = dataset._normalize_instruction_example(data[0])
    prompt_len = len(tokenizer.encode(normalized["prompt"]))
    response_len = len(tokenizer.encode(normalized["response"])) + 1  # +EOS
    valid_len = prompt_len + response_len

    assert int(item["attention_mask"].sum().item()) == valid_len
    assert torch.all(item["loss_mask"][:prompt_len] == 0)
    assert torch.all(item["labels"][:prompt_len] == -100)
    assert torch.all(item["loss_mask"][prompt_len:valid_len] == 1)
    assert torch.all(item["labels"][prompt_len:valid_len] >= 0)


def test_preference_dataset_formats_ranked_responses() -> None:
    tokenizer = TinyTokenizer()
    data = [
        {
            "prompt": "Rank these answers",
            "responses": ["worse", "better"],
            "scores": [0.1, 0.9],
        }
    ]
    dataset = PreferenceDataset(data, tokenizer=tokenizer, max_length=96)

    item = dataset[0]
    expected_keys = {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "chosen_loss_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
        "rejected_loss_mask",
    }
    assert set(item) == expected_keys
    for key in expected_keys:
        assert item[key].shape == (96,)
        assert item[key].dtype == torch.long

    assert not torch.equal(item["chosen_input_ids"], item["rejected_input_ids"])
    assert torch.all(item["chosen_labels"][item["chosen_loss_mask"] == 0] == -100)
    assert torch.all(item["rejected_labels"][item["rejected_loss_mask"] == 0] == -100)


def test_chat_dataset_formats_message_conversation() -> None:
    tokenizer = TinyTokenizer()
    data = [
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Ping"},
                {"role": "assistant", "content": "Pong"},
            ]
        }
    ]
    dataset = ChatDataset(data, tokenizer=tokenizer, max_length=96)

    item = dataset[0]
    _assert_standard_item(item, max_length=96)

    normalized = dataset._normalize_chat_example(data[0])
    prompt_len = len(tokenizer.encode(normalized["prompt"]))
    response_len = len(tokenizer.encode(normalized["response"])) + 1  # +EOS
    valid_len = prompt_len + response_len

    assert int(item["attention_mask"].sum().item()) == valid_len
    assert torch.all(item["loss_mask"][:prompt_len] == 0)
    assert torch.all(item["labels"][:prompt_len] == -100)
    assert torch.all(item["loss_mask"][prompt_len:valid_len] == 1)
    assert torch.all(item["labels"][prompt_len:valid_len] >= 0)


def test_concatenated_sequence_dataset_packs_expected_length() -> None:
    examples = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    dataset = ConcatenatedSequenceDataset(
        examples=examples,
        max_length=4,
        eos_token_id=None,
        pad_token_id=0,
        drop_remainder=False,
    )

    assert len(dataset) == 3
    assert dataset[0]["input_ids"].tolist() == [1, 2, 3, 4]
    assert dataset[1]["input_ids"].tolist() == [5, 6, 7, 8]
    assert dataset[2]["input_ids"].tolist() == [9, 0, 0, 0]
    assert dataset[2]["attention_mask"].tolist() == [1, 0, 0, 0]
    assert dataset[2]["labels"].tolist() == [9, -100, -100, -100]
    assert dataset[2]["loss_mask"].tolist() == [1, 0, 0, 0]


def test_resolve_dataset_source_prefers_suggested_alias_over_unsupported_existing_path(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # Existing path with alias name but unsupported local format should not block alias resolution.
    (tmp_path / "ultrachat").write_text("not-a-json-dataset", encoding="utf-8")

    resolved = resolve_dataset_source("ultrachat")

    assert resolved["source_type"] == "suggested"
    assert resolved["canonical_name"] == "ultrachat_200k"
    assert "HuggingFaceH4/ultrachat_200k" in resolved["hf_candidates"]
