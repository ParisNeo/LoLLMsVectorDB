from pathlib import Path
from typing import List
import tiktoken

class Tokenizer:
    def __init__(self):
        print("Base Tokenizer initialized. It's like a blank canvas, waiting for some color!")

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes the input text into a list of tokens.
        
        Args:
            text (str): The text to tokenize.
        
        Returns:
            List[int]: A list of tokens.
        """
        raise NotImplementedError("Tokenize method not implemented. It's like trying to drive a car without wheels!")

    def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenizes a list of tokens back into text.
        
        Args:
            tokens (List[int]): The list of tokens to detokenize.
        
        Returns:
            str: The detokenized text.
        """
        raise NotImplementedError("Detokenize method not implemented. It's like trying to bake a cake without an oven!")
