from typing import List
import tiktoken
from pathlib import Path
from lollmsvectordb.tokenizer import Tokenizer
from lollmsvectordb.lollms_tokenizers.tiktoken_tokenizer import TikTokenTokenizer
from lollmsvectordb.database_elements.document import Document
from lollmsvectordb.database_elements.chunk import Chunk

class TextChunker:
    def __init__(self, chunk_size: int = 512, tokenizer:Tokenizer=None):
        self.chunk_size = chunk_size
        if tokenizer is None:
            self.tokenizer = TikTokenTokenizer()

    def remove_unnecessary_returns(self, paragraph: str) -> str:
        """
        Removes unnecessary line returns (more than two) from a given paragraph.

        Args:
            paragraph (str): The input paragraph with potential unnecessary line returns.

        Returns:
            str: The paragraph with unnecessary line returns removed.
        """
        # Split the paragraph into lines
        lines = paragraph.splitlines()
        
        # Filter out empty lines and join with a single line return
        cleaned_paragraph = '\n'.join(line for line in lines if line.strip())
        
        return cleaned_paragraph

    def get_text_chunks(self, text: str, doc:Document, clean_chunk=True, min_nb_tokens_in_chunk=10) -> List[Chunk]:
        paragraphs = text.split('\n\n')  # Split text into paragraphs
        chunks = []
        current_chunk = []

        current_tokens = 0
        for paragraph in paragraphs:
            if clean_chunk:
                paragraph = paragraph.strip()
            paragraph_tokens = len(self.tokenizer.tokenize(paragraph))
            if current_tokens + paragraph_tokens > self.chunk_size:
                if current_tokens>min_nb_tokens_in_chunk:
                    chunk = Chunk(doc, b'', '\n\n'.join(current_chunk), current_tokens)
                    if clean_chunk:
                        chunk.text = self.remove_unnecessary_returns(chunk.text)
                    chunks.append(chunk)
                current_chunk = [paragraph]
                current_tokens = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens

        if current_chunk and current_tokens>min_nb_tokens_in_chunk:
            chunk = Chunk(doc.id, b'', '\n\n'.join(current_chunk), current_tokens)
            chunks.append(chunk)

        return chunks

# Example usage
if __name__ == "__main__":
    text_chunker = TextChunker(chunk_size=512)
    text = Path("test_file.txt").read_text()
    chunks = text_chunker._get_text_chunks(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")
