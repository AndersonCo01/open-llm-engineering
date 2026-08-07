"""Day 2: a character-level tokenizer built from first principles."""

from collections.abc import Iterable


class CharacterTokenizer:
    """Convert characters to integer token IDs and back again.

    Token ID 0 is reserved for characters that were not present in the training
    text. Known characters receive deterministic IDs starting at 1.
    """

    unknown_token = "<UNK>"
    unknown_token_id = 0

    def __init__(self, training_text: str) -> None:
        """Build a vocabulary containing every unique training character."""
        if not training_text:
            raise ValueError("Training text cannot be empty")
        
        unique_characters = sorted(set(training_text))

        self.token_to_id = {self.unknown_token: self.unknown_token_id}

        for token_id, character in enumerate(unique_characters, start=1):
            self.token_to_id[character] = token_id

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()

        }


    @property
    def vocabulary_size(self) -> int:
        """Return the number of known tokens plus the unknown token."""
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs, using ID 0 for unknown characters."""
        token_ids = []

        for character in text:
            token_id = self.token_to_id.get(
                character,
                self.unknown_token_id,
            )
            token_ids.append(token_id)

        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs to text, using <UNK> for unknown IDs."""
        tokens = []

        for token_id in token_ids:
            token = self.id_to_token.get(
                token_id,
                self.unknown_token,
            )
            tokens.append(token)

        return "".join(tokens)
            

if __name__ == "__main__":
    tokenizer = CharacterTokenizer("hello world")
    encoded = tokenizer.encode("hello!")
    print(f"vocabulary: {tokenizer.token_to_id}")
    print(f"encoded: {encoded}")
    print(f"decoded: {tokenizer.decode(encoded)}")

