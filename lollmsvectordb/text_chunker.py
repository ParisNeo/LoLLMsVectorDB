from typing import List
import tiktoken
from pathlib import Path

class TextChunker:
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _get_text_chunks(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')  # Split text into paragraphs
        chunks = []
        current_chunk = []

        current_tokens = 0
        for paragraph in paragraphs:
            paragraph_tokens = len(self.tokenizer.encode(paragraph))
            if current_tokens + paragraph_tokens > self.chunk_size:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

# Example usage
if __name__ == "__main__":
    text_chunker = TextChunker(chunk_size=512)
    text = Path("test_file.txt").read_text()
    chunks = text_chunker._get_text_chunks(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")
