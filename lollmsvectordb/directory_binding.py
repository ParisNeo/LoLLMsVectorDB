"""
LoLLMsVectorDB

File: directory_binding.py
Author: ParisNeo
Description: Contains the DirectoryBinding class for managing text data from a specified directory.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

import hashlib
import os

from lollmsvectordb.text_chunker import TextChunker
from lollmsvectordb.text_document_loader import TextDocumentsLoader
from lollmsvectordb.vector_database import VectorDatabase


class DirectoryBinding:
    """
    Binds a vector database to a folder for ease-of-use
    """
    def __init__(self, directory_path: str, vector_database: VectorDatabase, chunk_size: int=512):
        """Initialize the binding
        
        Args:
            directory_path: Path to the directory to bind
            vector_database: The vector database object to associate the folder with
            chunk_size: The vectors chunk size
        """
        self.directory_path = directory_path
        self.vector_database = vector_database
        self.file_hashes = {}
        self.text_chunker = TextChunker(chunk_size)
        self.text_loader = TextDocumentsLoader()

    def _hash_file(self, file_path):
        """Returns a hash of the file contents
        
        Args: 
            file_path: Path to the file to hash
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def update_vector_store(self):
        """Sweeps the binded directory and chunk its content, then update
        the vector database binded with the directory for retrieval.
        """
        current_files = set()
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                current_files.add(file_path)
                file_hash = self._hash_file(file_path)

                if (
                    file_path not in self.file_hashes
                    or self.file_hashes[file_path] != file_hash
                ):
                    self.file_hashes[file_path] = file_hash
                    
                    text, doc = self.text_loader.load_document(file_path)

                    title = os.path.basename(file_path)
                    self.vector_database.add_document(
                        title=title,
                        text=text,
                        path=file_path,
                        force_update=True
                    )

        # Remove vectors for files that no longer exist
        removed_files = set(self.file_hashes.keys()) - current_files
        for removed_file in removed_files:
            self.vector_database.remove_vectors_by_meta_prefix(removed_file)
            del self.file_hashes[removed_file]

        self.vector_database.build_index()

    def search(self, query: str, n_results: int=5):
        """Searches the n most relevant context pieces for a given query
        
        Args:
            query: The search query
            n_results: The number of closest contexts to retrieve
        """
        chunks = self.vector_database.search(query, n_results)

        results = []
        for chunk in chunks:
            results.append((chunk.doc.path, chunk.chunk_id, chunk.distance, chunk.text))

        return results
