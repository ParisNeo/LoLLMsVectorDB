from pathlib import Path
from typing import List

import tiktoken
from ascii_colors import ASCIIColors

from lollmsvectordb.tokenizer import Tokenizer
from datetime import datetime

class TikTokenTokenizer(Tokenizer):
    def __init__(self, encoding="gpt2"):
        super().__init__("TiktokenTokenizer")
        self.tokenizer = tiktoken.get_encoding(encoding)
        ASCIIColors.multicolor(
            [f"[LollmsVectorDB][{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", "TikTokenTokenizer initialized."],
            [ASCIIColors.color_red, ASCIIColors.color_green],
        )

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes the input text into a list of tokens.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[int]: A list of tokens.
        """
        tokens = self.tokenizer.encode(text)
        return tokens

    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenizes a list of tokens back into text.

        Args:
            tokens (List[int]): The list of tokens to detokenize.

        Returns:
            str: The detokenized text.
        """
        text = self.tokenizer.decode(tokens)
        return text
