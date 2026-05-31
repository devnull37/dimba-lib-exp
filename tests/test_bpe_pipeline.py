import pytest
pytest.importorskip("tokenizers")

from dimba.tokenizers.bpe import BPETokenizer

TEXTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "machine learning is transforming the world",
    "byte pair encoding is a subword tokenization algorithm",
    "natural language processing with neural networks",
    "the cat sat on the mat",
    "transformers are powerful sequence models",
    "training on large corpora yields better representations",
]


@pytest.fixture(scope="module")
def trained_tokenizer():
    tok = BPETokenizer.train(TEXTS, vocab_size=300)
    tok.vocab_size = tok.tokenizer.get_vocab_size()
    return tok


def test_train_returns_tokenizer(trained_tokenizer):
    assert isinstance(trained_tokenizer, BPETokenizer)


def test_vocab_size_positive(trained_tokenizer):
    assert isinstance(trained_tokenizer.vocab_size, int)
    assert trained_tokenizer.vocab_size > 0


def test_encode_returns_list_of_int(trained_tokenizer):
    ids = trained_tokenizer.encode("hello world")
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_decode_returns_str(trained_tokenizer):
    ids = trained_tokenizer.encode("hello world")
    text = trained_tokenizer.decode(ids)
    assert isinstance(text, str)


def test_round_trip_nonempty(trained_tokenizer):
    sample = "machine learning"
    ids = trained_tokenizer.encode(sample)
    result = trained_tokenizer.decode(ids)
    assert isinstance(result, str)
    assert len(result) > 0
