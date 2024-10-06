from typing import List, Optional
from pathlib import Path
from lollmsvectordb.tokenizer import Tokenizer
from lollmsvectordb.lollms_tokenizers.tiktoken_tokenizer import TikTokenTokenizer
from lollmsvectordb.database_elements.document import Document
from lollmsvectordb.database_elements.chunk import Chunk
from lollmsvectordb.llm_model import LLMModel
import re
class TextChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 0, tokenizer: Optional[Tokenizer] = None, model: Optional[LLMModel] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer if tokenizer else TikTokenTokenizer()
        self.model = model



    def get_text_chunks(self, text: str, doc: Document, clean_chunk: bool = True, min_nb_tokens_in_chunk: int = 1) -> List[Chunk]:
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
        paragraph_tokens = len(self.tokenizer.tokenize(text))
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
                        chunk_text = TextChunker.remove_unnecessary_returns(chunk_text)
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
                chunk_text = TextChunker.remove_unnecessary_returns(chunk_text)
            chunk = Chunk(doc, b'', chunk_text, current_tokens, chunk_id)
            chunks.append(chunk)

        return chunks
    @staticmethod
    def remove_unnecessary_returns(paragraph: str) -> str:
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
    
    @staticmethod
    def chunk_text(text: str, tokenizer:Tokenizer, chunk_size=512, overlap=0, clean_chunk: bool = True, min_nb_tokens_in_chunk: int = 10) -> List[str]:
        """
        Splits the input text into chunks based on the specified chunk size and overlap.

        Args:
            text (str): The input text to be chunked.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for counting tokens.
            chunk_size (int): The maximum number of tokens per chunk.
            overlap (int): The number of overlapping tokens between chunks.
            clean_chunk (bool): Whether to clean the chunk by removing unnecessary returns.
            min_nb_tokens_in_chunk (int): The minimum number of tokens required in a chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        def split_paragraph(paragraph: str) -> List[str]:
            sentences = re.split('(?<=[.!?]) +', paragraph)
            split_sentences = []
            current_sentence = ""
            for sentence in sentences:
                if len(tokenizer.tokenize(current_sentence + sentence)) > chunk_size:
                    if current_sentence:
                        split_sentences.append(current_sentence.strip())
                    words = sentence.split()
                    current_sentence = ""
                    for word in words:
                        if len(tokenizer.tokenize(current_sentence + word)) > chunk_size:
                            if current_sentence:
                                split_sentences.append(current_sentence.strip())
                            current_sentence = word + " "
                        else:
                            current_sentence += word + " "
                else:
                    current_sentence += sentence + " "
            if current_sentence:
                split_sentences.append(current_sentence.strip())
            return split_sentences

        paragraphs = text.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            if clean_chunk:
                paragraph = paragraph.strip()
            
            split_paragraphs = split_paragraph(paragraph)
            
            for split_para in split_paragraphs:
                para_tokens = len(tokenizer.tokenize(split_para))
                
                if current_tokens + para_tokens > chunk_size:
                    if current_tokens > min_nb_tokens_in_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if clean_chunk:
                            chunk_text = TextChunker.remove_unnecessary_returns(chunk_text)
                        chunks.append(chunk_text)
                    
                    if overlap > 0:
                        overlap_tokens = tokenizer.tokenize(' '.join(current_chunk[-overlap:]))
                        current_chunk = [tokenizer.decode(overlap_tokens)]
                        current_tokens = len(overlap_tokens)
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(split_para)
                current_tokens += para_tokens

        if current_chunk and current_tokens > min_nb_tokens_in_chunk:
            chunk_text = ' '.join(current_chunk)
            if clean_chunk:
                chunk_text = TextChunker.remove_unnecessary_returns(chunk_text)
            chunks.append(chunk_text)

        return chunks
# Example usage
if __name__ == "__main__":
    text_chunker = TextChunker(chunk_size=512, overlap=50)
    text = Path("test_file.txt").read_text()
    doc = Document(id=1, content=text)
    chunks = text_chunker.get_text_chunks(text, doc)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk.text}\n")
