"""
LoLLMsVectorDB

File: vector_database.py
Author: ParisNeo
Description: Contains the VectorDatabase class for managing and searching vectorized text data.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

import numpy as np
import sqlite3
import hashlib
from sklearn.neighbors import NearestNeighbors
from lollmsvectordb.vectorizer import Vectorizer
from typing import List, Tuple, Optional
from ascii_colors import ASCIIColors
class VectorDatabase:
    """
    A class to manage a vector database using SQLite and perform nearest neighbor searches.

    Attributes:
    -----------
    db_path : str
        Path to the SQLite database file.
    vectorizer : Vectorizer
        An instance of a vectorizer to convert data into vectors.
    n_neighbors : int, optional
        Number of neighbors to use for k-neighbors queries (default is 5).
    algorithm : str, optional
        Algorithm to use for nearest neighbors search (default is 'auto').
    nn_model : NearestNeighbors or None
        The nearest neighbors model, initialized in build_index().
    vectors : list
        List of vectors loaded from the database.
    text : list
        List of text corresponding to the vectors.
    """

    def __init__(self, db_path: str, vectorizer: Vectorizer, n_neighbors: int = 5, algorithm: str = 'auto', metrics:str="euclidean"):
        """
        Initializes the VectorDatabase with the given parameters.

        Parameters:
        -----------
        db_path : str
            Path to the SQLite database file.
        vectorizer : Vectorizer
            An instance of a vectorizer to convert data into vectors.
        n_neighbors : int, optional
            Number of neighbors to use for k-neighbors queries (default is 5).
        algorithm : str, optional
            Algorithm to use for nearest neighbors search (default is 'auto').
            'auto': This will attempt to decide the most appropriate algorithm based on the values passed to fit method.
            'ball_tree': This algorithm uses a Ball Tree data structure. It is efficient for low-dimensional data.
            'kd_tree': This algorithm uses a KD Tree data structure. It is efficient for low-dimensional data.
            'brute': This algorithm performs a brute-force search. It is efficient for high-dimensional data and when the dataset is small.
            'hnsw': This algorithm uses Hierarchical Navigable Small World graphs. It is efficient for high-dimensional data and large datasets. Note that this algorithm is available in scikit-learn from version 1.0.0 onwards.
        metrics : str, optional
            Euclidean Distance ('euclidean'): The straight-line distance between two points in Euclidean space.
            Manhattan Distance ('manhattan'): Also known as L1 distance or city block distance, it is the sum of the absolute differences of their coordinates.
            Chebyshev Distance ('chebyshev'): The maximum distance along any coordinate dimension.
            Minkowski Distance ('minkowski'): A generalization of Euclidean and Manhattan distances. It is defined by a parameter p, where p=1 is equivalent to Manhattan distance and p=2 is equivalent to Euclidean distance.
            Cosine Distance ('cosine'): Measures the cosine of the angle between two vectors. It is often used for text data.
            Hamming Distance ('hamming'): Measures the proportion of differing components between two binary vectors.
            Jaccard Distance ('jaccard'): Measures the dissimilarity between two sets. It is the complement of the Jaccard similarity coefficient.
            Mahalanobis Distance ('mahalanobis'): Measures the distance between a point and a distribution. It accounts for the correlations of the data set.
            Canberra Distance ('canberra'): A weighted version of Manhattan distance.
            Bray-Curtis Distance ('braycurtis'): Measures the dissimilarity between two vectors.
        """
        self.db_path = db_path
        self.vectorizer = vectorizer
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metrics    = metrics
        self.nn_model = None
        self.vectors = []
        self.text = []

        self._create_tables()

    def _create_tables(self):
        """
        Creates the necessary tables in the SQLite database if they do not exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    hash TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER,
                    vector BLOB NOT NULL,
                    text TEXT NOT NULL,
                    title TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vectorizer_info (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
            ''')
            conn.commit()

    def _hash_document(self, text: str) -> str:
        """
        Generates a SHA-256 hash for the given text.

        Parameters:
        -----------
        text : str
            The text to be hashed.

        Returns:
        --------
        str
            The SHA-256 hash of the text.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get_first_vectorizer_name(self) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM vectorizer_info ORDER BY id LIMIT 1')
            result = cursor.fetchone()
            return result[0] if result else ""
    
    def add_document(self, title: str, text: str, chunk_size: int = 512):
        """
        Adds a document and its chunks to the database.

        Parameters:
        -----------
        title : str
            The title of the document.
        text : str
            The full text of the document.
        chunk_size : int, optional
            The size of each chunk (default is 512).
        """
        doc_hash = self._hash_document(text)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM documents WHERE hash = ?
            ''', (doc_hash,))
            result = cursor.fetchone()
            
            if result is not None:
                print(f"Document with hash {doc_hash} already exists.")
                return
            
            cursor.execute('''
                INSERT INTO documents (hash, title) VALUES (?, ?)
            ''', (doc_hash, title))
            document_id = cursor.lastrowid

            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            for chunk in chunks:
                vector = self.vectorizer.vectorize([chunk])[0]
                vector_blob = np.array(vector).tobytes()
                cursor.execute('''
                    INSERT INTO chunks (document_id, vector, text, title) VALUES (?, ?, ?, ?)
                ''', (document_id, vector_blob, chunk, title))
            conn.commit()

    def remove_document(self, doc_hash: str):
        """
        Removes a document and its chunks from the database.

        Parameters:
        -----------
        doc_hash : str
            The hash of the document to be removed.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM chunks WHERE document_id IN (
                    SELECT id FROM documents WHERE hash = ?
                )
            ''', (doc_hash,))
            cursor.execute('''
                DELETE FROM documents WHERE hash = ?
            ''', (doc_hash,))
            conn.commit()

    def verify_document(self, text: str) -> bool:
        """
        Verifies if a document exists in the database by its hash.

        Parameters:
        -----------
        text : str
            The full text of the document to be verified.

        Returns:
        --------
        bool
            True if the document exists, False otherwise.
        """
        doc_hash = self._hash_document(text)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 1 FROM documents WHERE hash = ?
            ''', (doc_hash,))
            return cursor.fetchone() is not None

    def _load_vectors(self):
        """
        Loads vectors and their text from the database into memory.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT chunks.vector, chunks.text, documents.title 
                FROM chunks 
                JOIN documents ON chunks.document_id = documents.id
            ''')
            rows = cursor.fetchall()
            self.vectors = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
            self.texts = [row[1] for row in rows]
            self.titles = [row[2] for row in rows]

    def _update_vectors(self):
        """
        Updates vectors in the database using the current vectorizer.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, text FROM chunks')
            rows = cursor.fetchall()
            for row in rows:
                chunk_id, text = row
                vector = self.vectorizer.vectorize([text])[0]
                vector_blob = vector.tobytes()
                cursor.execute('UPDATE chunks SET vector = ? WHERE id = ?', (vector_blob, chunk_id))
            conn.commit()

    def build_index(self):
        """
        Builds the nearest neighbors index using the loaded vectors.
        """
        if self.vectorizer.name == self.get_first_vectorizer_name():
            ASCIIColors.warning("Detected a change in the vectorizer. Revectorizing the whole database")
            self._update_vectors()
        else:
            self._load_vectors()
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric='cosine')
        self.nn_model.fit(self.vectors)

    def search(self, query_data: str, n_results: int = 5) -> List[Tuple[np.ndarray, str, float]]:
        """
        Searches for the nearest neighbors of the query data.

        Parameters:
        -----------
        query_data : str
            The data to be vectorized and searched in the database.
        n_results : int, optional
            Number of nearest neighbors to return (default is 5).

        Returns:
        --------
        list of tuples
            A list of tuples containing the vector, text, and distance of the nearest neighbors.
        """
        if self.nn_model is None:
            raise ValueError("Index not built. Call build_index() before searching.")
        
        if len(self.vectors) < n_results:
            n_results=len(self.vectors)
        
        query_vector = self.vectorizer.vectorize([query_data])[0]
        distances, indices = self.nn_model.kneighbors([query_vector], n_neighbors=n_results)
        results = [(self.vectors[idx], self.texts[idx], self.titles[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results

    def save(self):
        """
        Placeholder method for saving the database. SQLite handles persistence automatically.
        """
        pass

    def load(self):
        """
        Loads the vectors and builds the nearest neighbors index.
        """
        self.build_index()

    def remove_vectors_by_meta_prefix(self, meta_prefix: str):
        """
        Removes vectors from the database whose text starts with the given prefix.

        Parameters:
        -----------
        meta_prefix : str
            The prefix of the text to match for deletion.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM chunks WHERE text LIKE ?
            ''', (f"{meta_prefix}%",))
            conn.commit()


# Example usage
if __name__ == "__main__":
    # Example with TFIDFVectorizer
    from lollmsvectordb import TFIDFVectorizer
    from lollmsvectordb.vectorizers.bert_vectorizer import BERTVectorizer
    tfidf_vectorizer = BERTVectorizer()
    tfidf_vectorizer.fit(["This is a sample text.", "Another sample text.", "Another sample text 1.", "Another sample text 2.", "Another sample text 3.", "Another sample text 4."])
    db = VectorDatabase("vector_db.sqlite", tfidf_vectorizer)

    # Add multiple documents to the database
    documents = [
        ("Document 1", "This is the first sample text."),
        ("Document 2", "Here is another example of a sample text."),
        ("Document 3", "This document is different from the others."),
        ("Document 4", "Yet another document with some sample text."),
        ("Document 5", "This is the fifth document in the database."),
        ("Document 6", "Finally, this is the sixth sample text.")
    ]

    for title, text in documents:
        db.add_document(title, text)

    # Build the nearest neighbors index
    db.build_index()

    # Perform a search query
    query = "what is the sixth sample text."
    results = db.search(query, n_results=3)

    # Print the search results
    for vector, texts, titles, distance in results:
        print(f"Title: {title}, Text: {text}, Distance: {distance}")
