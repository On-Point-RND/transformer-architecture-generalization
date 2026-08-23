import string


class CharTokenizer:
    """Character-level tokenizer for algorithmic sequence modeling.

    Maps printable ASCII characters and special control tokens (<pad>, <bos>, <eos>)
    to unique integer IDs and reconstructs strings from token ID sequences.
    """

    def __init__(self):
        """Initializes the tokenizer and builds vocabulary mapping dictionaries.

        Reserves IDs 0, 1, and 2 for special tokens <pad>, <bos>, and <eos>,
        shifting printable character IDs by 3 to avoid collision.
        """
        chars = list(string.printable)

        self.PAD_ID = 0
        self.BOS_ID = 1  # (Begin of Sequence)
        self.EOS_ID = 2  # (End of Sequence)
        self.c2i = {c: i + 3 for i, c in enumerate(chars)}
        self.c2i["<pad>"] = self.PAD_ID
        self.c2i["<bos>"] = self.BOS_ID
        self.c2i["<eos>"] = self.EOS_ID
        self.i2c = {i: c for c, i in self.c2i.items()}

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        return len(self.c2i)

    def encode(self, text: str) -> list[int]:
        """Encodes an input string into a list of token IDs.

        Args:
            text: The input string to be tokenized.

        Returns:
            A list of integer token IDs.
        """
        return [self.c2i[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string representation.

        Control and padding tokens (IDs 0, 1, and 2) are automatically
        filtered out during reconstruction.

        Args:
            ids: A sequence of integer token IDs.

        Returns:
            The reconstructed string.
        """
        return "".join(self.i2c[i] for i in ids if i > 2)