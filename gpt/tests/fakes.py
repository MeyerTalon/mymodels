PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
FIRST_CHAR_ID = 3
ALPHABET = 'abcdefghijklmnopqrstuvwxyz ,.'


class CharTokenizer:
    vocab_size = FIRST_CHAR_ID + len(ALPHABET)
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    def encode(self, text: str) -> list[int]:
        return [FIRST_CHAR_ID + ALPHABET.index(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return ''.join(
            ALPHABET[token_id - FIRST_CHAR_ID]
            for token_id in token_ids
            if token_id >= FIRST_CHAR_ID
        )
