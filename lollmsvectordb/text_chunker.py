from typing import List, Optional
from pathlib import Path
from lollmsvectordb.tokenizer import Tokenizer
from lollmsvectordb.lollms_tokenizers.tiktoken_tokenizer import TikTokenTokenizer
from lollmsvectordb.database_elements.document import Document
from lollmsvectordb.database_elements.chunk import Chunk
from lollmsvectordb.llm_model import LLMModel

class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 0, tokenizer: Optional[Tokenizer] = None, model: Optional[LLMModel] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer if tokenizer else TikTokenTokenizer()
        self.model = model

    def remove_unnecessary_returns(self, paragraph: str) -> str:
        """
        Removes unnecessary line returns (more than two) from a given paragraph.

        Args:
            paragraph (str): The input paragraph with potential unnecessary line returns.

        Returns:
            str: The paragraph with unnecessary line returns removed.
        """
        lines = paragraph.splitlines()
        cleaned_paragraph = '\n'.join(line for line in lines if line.strip())
        return cleaned_paragraph

    def get_text_chunks(self, text: str, doc: Document, clean_chunk: bool = True, min_nb_tokens_in_chunk: int = 10) -> List[Chunk]:
        """
        Splits the input text into chunks based on the specified chunk size and overlap.

        Args:
            text (str): The input text to be chunked.
            doc (Document): The document object associated with the chunks.
            clean_chunk (bool): Whether to clean the chunk by removing unnecessary returns.
            min_nb_tokens_in_chunk (int): The minimum number of tokens required in a chunk.

        Returns:
            List[Chunk]: A list of Chunk objects.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for paragraph in paragraphs:
            if clean_chunk:
                paragraph = paragraph.strip()
            paragraph_tokens = len(self.tokenizer.tokenize(paragraph))
            if current_tokens + paragraph_tokens > self.chunk_size:
                if current_tokens > min_nb_tokens_in_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if clean_chunk:
                        chunk_text = self.remove_unnecessary_returns(chunk_text)
                    if self.model:
                        chunk_text = self.model.generate(chunk_text, self.chunk_size)
                    chunk = Chunk(doc, b'', chunk_text, current_tokens, chunk_id=chunk_id)
                    chunk_id += 1
                    chunks.append(chunk)
                if self.overlap > 0:
                    current_chunk = current_chunk[-self.overlap:] + [paragraph]
                else:
                    current_chunk = [paragraph]
                current_tokens = sum(len(self.tokenizer.tokenize(p)) for p in current_chunk)
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens

        if current_chunk and current_tokens > min_nb_tokens_in_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if clean_chunk:
                chunk_text = self.remove_unnecessary_returns(chunk_text)
            if self.model:
                chunk_text = self.model.generate(chunk_text, self.chunk_size)
            chunk = Chunk(doc, b'', chunk_text, current_tokens, chunk_id)
            chunks.append(chunk)

        return chunks
# Example usage
if __name__ == "__main__":
    text_chunker = TextChunker(chunk_size=512, overlap=50)
    text = Path("test_file.txt").read_text()
    doc = Document(id=1, content=text)
    chunks = text_chunker.get_text_chunks(text, doc)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk.text}\n")
