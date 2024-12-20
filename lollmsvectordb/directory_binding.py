"""
LoLLMsVectorDB

File: directory_binding.py
Author: ParisNeo
Description: Contains the DirectoryBinding class for managing text data from a specified directory.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

import hashlib
import os
import sqlite3

from lollmsvectordb.text_chunker import TextChunker
from lollmsvectordb.text_document_loader import TextDocumentsLoader
from lollmsvectordb.vector_database import VectorDatabase


class DirectoryBinding:
    def __init__(self, directory_path, vector_database: VectorDatabase, chunk_size=512):
        self.directory_path = directory_path
        self.vector_database = vector_database
        self.file_hashes = {}
        self.text_chunker = TextChunker(chunk_size)
        self.text_loader = TextDocumentsLoader()

    def _hash_file(self, file_path):
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def update_vector_store(self):
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
                    text = self.text_loader.read_file(file)
                    chunks = self.text_chunker._get_text_chunks(text)
                    for i, chunk in enumerate(chunks):
                        meta = f"{file_path}:{i}"
                        self.vector_database.add_vector(chunk, meta)

        # Remove vectors for files that no longer exist
        removed_files = set(self.file_hashes.keys()) - current_files
        for removed_file in removed_files:
            self.vector_database.remove_vectors_by_meta_prefix(removed_file)
            del self.file_hashes[removed_file]

        self.vector_database.build_index()

    def search(self, query, n_results=5):
        results = self.vector_database.search(query, n_results)
        return [
            (meta.split(":")[0], int(meta.split(":")[1]), distance)
            for _, meta, distance in results
        ]
