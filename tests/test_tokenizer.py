import pytest

from open_llm_engineering.tokenizer import CharacterTokenizer


def test_empty_training_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        CharacterTokenizer("")


def test_vocabulary_is_unique_and_includes_unknown_token() -> None:
    tokenizer = CharacterTokenizer("banana")

    assert tokenizer.vocabulary_size == 4
    assert tokenizer.token_to_id == {"<UNK>": 0, "a": 1, "b": 2, "n": 3}


def test_known_text_round_trip() -> None:
    tokenizer = CharacterTokenizer("hello world")
    original = "hello"

    assert tokenizer.decode(tokenizer.encode(original)) == original


def test_unknown_character_uses_reserved_token() -> None:
    tokenizer = CharacterTokenizer("abc")

    assert tokenizer.encode("a?") == [1, 0]
    assert tokenizer.decode([1, 0]) == "a<UNK>"


def test_unknown_integer_id_decodes_safely() -> None:
    tokenizer = CharacterTokenizer("abc")

    assert tokenizer.decode([999]) == "<UNK>"

