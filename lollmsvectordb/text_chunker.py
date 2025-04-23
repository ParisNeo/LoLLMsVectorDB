import re
from pathlib import Path
from typing import List, Optional, Tuple

from lollmsvectordb.database_elements.chunk import Chunk
from lollmsvectordb.database_elements.document import Document
from lollmsvectordb.llm_model import LLMModel
from lollmsvectordb.lollms_tokenizers.tiktoken_tokenizer import \
    TikTokenTokenizer
from lollmsvectordb.tokenizer import Tokenizer


class TextChunker:
    """
    The Text Chunker provides methods to chunk texts, with parametrized
    chunk sizes, overlap and optionnal LLM & Tokenizer support.
    """
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 0,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[LLMModel] = None,
    ) -> None:
        """
        Initializes the Text Chunker
        
        Args:
            chunk_size: The text chunks size (in tokens)
            overlap: The overlap between two adjacent chunks (in tokens)
            tokenizer: The tokenizer used to tokenize text
            model: The model (unused)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer if tokenizer else TikTokenTokenizer()
        self.model = model

    def split_into_sentences(self, paragraph):
        """
        Split a paragraph into sentences.

        Args:
            paragraph (str): The text to be split into sentences

        Returns:
            list: A list of sentences
        """
        if not paragraph:
            return []

        import re

        # Define abbreviations to avoid false splits
        abbreviations = (
            r"(?:[A-Za-z]\.){2,}|Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|etc\."
        )

        # Define sentence endings
        sentence_endings = r"[.!?]+"

        # Define quotes and parentheses
        quotes_parentheses = r'["\'\(\)\[\]\{\}]'

        # Split the paragraph into sentences
        # First, replace abbreviations with a placeholder to avoid false splits
        text = paragraph
        abbrev_matches = re.finditer(abbreviations, text)
        placeholders = {}
        for i, match in enumerate(abbrev_matches):
            placeholder = f"ABBREV_{i}"
            placeholders[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)

        # Split on sentence endings followed by spaces and capital letters
        pattern = f"({sentence_endings}+)\\s+(?=[A-Z]|{quotes_parentheses}\\s*[A-Z])"
        sentences = re.split(pattern, text)

        # Rejoin sentence endings with their sentences
        sentences = [
            "".join(pair) for pair in zip(sentences[::2], sentences[1::2] + [""])
        ]

        # Remove empty strings and whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        # Restore abbreviations
        for sentence in sentences:
            for placeholder, abbrev in placeholders.items():
                sentence = sentence.replace(placeholder, abbrev)

        return sentences

    def get_text_chunks(
        self,
        text: str,
        doc: Document,
        clean_chunk: bool = True,
        min_nb_tokens_in_chunk: int = 1,
    ) -> List[Chunk]:
        """Chunks the input text and stores it in a list.
        
        Args:
            text: The input text to chunk.
            doc: The document to associate the input text with.
            clean_chunk: Whether to clean each paragraph from text before
                chunking.
            min_nb_tokens_in_chunk: Minimum number of tokens in a chunk.
        """
        paragraphs = text.split("\n")
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for paragraph in paragraphs:
            if clean_chunk:
                paragraph = paragraph.strip()
            paragraph_tokens = len(self.tokenizer.tokenize(paragraph))

            if paragraph_tokens > self.chunk_size:
                # Handle large paragraphs
                sentences = self.split_into_sentences(paragraph)
                for sentence in sentences:
                    sentence_tokens = len(self.tokenizer.tokenize(sentence))
                    if sentence_tokens > self.chunk_size:
                        # Split large sentences
                        words = sentence.split()
                        for i in range(0, len(words), self.chunk_size):
                            sub_sentence = " ".join(words[i : i + self.chunk_size])
                            sub_sentence_tokens = len(
                                self.tokenizer.tokenize(sub_sentence)
                            )
                            self.add_to_chunks(
                                sub_sentence,
                                sub_sentence_tokens,
                                current_chunk,
                                current_tokens,
                                chunks,
                                doc,
                                clean_chunk,
                                min_nb_tokens_in_chunk,
                            )
                            current_chunk, current_tokens = self.reset_chunk(
                                current_chunk, self.overlap
                            )
                            chunk_id += 1
                    else:
                        self.add_to_chunks(
                            sentence,
                            sentence_tokens,
                            current_chunk,
                            current_tokens,
                            chunks,
                            doc,
                            clean_chunk,
                            min_nb_tokens_in_chunk,
                        )
                        current_chunk, current_tokens = self.reset_chunk(
                            current_chunk, self.overlap
                        )
                        chunk_id += 1
            elif current_tokens + paragraph_tokens > self.chunk_size:
                # Current chunk is full, start a new one
                self.add_to_chunks(
                    "\n\n".join(current_chunk),
                    current_tokens,
                    current_chunk,
                    current_tokens,
                    chunks,
                    doc,
                    clean_chunk,
                    min_nb_tokens_in_chunk,
                )
                current_chunk, current_tokens = self.reset_chunk(
                    current_chunk, self.overlap
                )
                chunk_id += 1
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens

        # Add any remaining content
        if current_chunk and current_tokens > min_nb_tokens_in_chunk:
            self.add_to_chunks(
                "\n\n".join(current_chunk),
                current_tokens,
                current_chunk,
                current_tokens,
                chunks,
                doc,
                clean_chunk,
                min_nb_tokens_in_chunk,
            )

        return chunks

    def add_to_chunks(
        self,
        text: str,
        tokens: int,
        current_chunk: List[str],
        current_tokens: int,
        chunks: List[Chunk],
        doc: Document,
        clean_chunk: bool,
        min_nb_tokens_in_chunk: int,
    ) -> None:
        """Adds a chunk to the chunk list"""
        if current_tokens > min_nb_tokens_in_chunk:
            chunk_text = "\n\n".join(current_chunk) if current_chunk else text
            if clean_chunk:
                chunk_text = self.remove_unnecessary_returns(chunk_text)
            chunk = Chunk(doc, b"", chunk_text, current_tokens, chunk_id=len(chunks))
            chunks.append(chunk)

    def reset_chunk(
        self, current_chunk: List[str], overlap: int
    ) -> Tuple[List[str], int]:
        """Resets the current chunk"""
        if overlap > 0 and current_chunk:
            overlap_elements = min(overlap, len(current_chunk))
            return current_chunk[-overlap_elements:], sum(
                len(self.tokenizer.tokenize(p)) for p in current_chunk[-overlap_elements:]
            )
        else:
            return [], 0

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
        cleaned_paragraph = "\n".join(line for line in lines if line.strip())
        return cleaned_paragraph

    @staticmethod
    def chunk_text(
        text: str,
        tokenizer: Tokenizer,
        chunk_size=512,
        overlap=0,
        clean_chunk: bool = True,
        min_nb_tokens_in_chunk: int = 10,
    ) -> List[str]:
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
            sentences = re.split("(?<=[.!?]) +", paragraph)
            split_sentences = []
            current_sentence = ""
            for sentence in sentences:
                if len(tokenizer.tokenize(current_sentence + sentence)) > chunk_size:
                    if current_sentence:
                        split_sentences.append(current_sentence.strip())

                    # Handle very long sentences
                    if len(tokenizer.tokenize(sentence)) > chunk_size:
                        words = sentence.split()
                        current_sentence = ""
                        for word in words:
                            if (
                                len(tokenizer.tokenize(current_sentence + word))
                                > chunk_size
                            ):
                                if current_sentence:
                                    split_sentences.append(current_sentence.strip())
                                current_sentence = word + " "
                            else:
                                current_sentence += word + " "
                    else:
                        current_sentence = sentence + " "
                else:
                    current_sentence += sentence + " "

            if current_sentence:
                split_sentences.append(current_sentence.strip())

            return split_sentences

        paragraphs = text.split("\n")
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
                        chunk_text = " ".join(current_chunk)
                        if clean_chunk:
                            chunk_text = TextChunker.remove_unnecessary_returns(
                                chunk_text
                            )
                        chunks.append(chunk_text)

                    if overlap > 0:
                        overlap_tokens = tokenizer.tokenize(
                            " ".join(current_chunk[-overlap:])
                        )
                        current_chunk = [tokenizer.decode(overlap_tokens)]
                        current_tokens = len(overlap_tokens)
                    else:
                        current_chunk = []
                        current_tokens = 0

                current_chunk.append(split_para)
                current_tokens += para_tokens

        if current_chunk and current_tokens > min_nb_tokens_in_chunk:
            chunk_text = " ".join(current_chunk)
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
