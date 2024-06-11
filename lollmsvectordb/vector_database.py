"""
LoLLMsVectorDB

File: vector_database.py
Author: ParisNeo
Description: Contains the VectorDatabase class for managing and searching vectorized text data.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

import numpy as np
import sqlite3
from sklearn.neighbors import NearestNeighbors
from lollmsvectordb.vectorizer import Vectorizer

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
    metadata : list
        List of metadata corresponding to the vectors.
    """

    def __init__(self, db_path: str, vectorizer: Vectorizer, n_neighbors: int = 5, algorithm: str = 'auto'):
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
        """
        self.db_path = db_path
        self.vectorizer = vectorizer
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.nn_model = None
        self.vectors = []
        self.metadata = []

        self._create_tables()

    def _create_tables(self):
        """
        Creates the necessary tables in the SQLite database if they do not exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY,
                    vector BLOB NOT NULL,
                    metadata TEXT
                )
            ''')
            conn.commit()

    def add_vector(self, data: str, meta: str = None):
        """
        Adds a vectorized representation of the data to the database.

        Parameters:
        -----------
        data : str
            The data to be vectorized and added to the database.
        meta : str, optional
            Metadata associated with the data (default is None).
        """
        vector = self.vectorizer.vectorize([data])[0]
        vector_blob = np.array(vector).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO vectors (vector, metadata) VALUES (?, ?)
            ''', (vector_blob, meta))
            conn.commit()

    def _load_vectors(self):
        """
        Loads vectors and their metadata from the database into memory.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT vector, metadata FROM vectors')
            rows = cursor.fetchall()
            self.vectors = [np.frombuffer(row[0], dtype=np.float64) for row in rows]
            self.metadata = [row[1] for row in rows]

    def build_index(self):
        """
        Builds the nearest neighbors index using the loaded vectors.
        """
        self._load_vectors()
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm)
        self.nn_model.fit(self.vectors)

    def search(self, query_data: str, n_results: int = 5):
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
            A list of tuples containing the vector, metadata, and distance of the nearest neighbors.
        """
        if self.nn_model is None:
            raise ValueError("Index not built. Call build_index() before searching.")
        
        if len(self.vectors) < n_results:
            raise ValueError(f"Expected n_neighbors <= n_samples, but n_samples = {len(self.vectors)}, n_neighbors = {n_results}")
        
        query_vector = self.vectorizer.vectorize([query_data])[0]
        distances, indices = self.nn_model.kneighbors([query_vector], n_neighbors=n_results)
        results = [(self.vectors[idx], self.metadata[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
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
        Removes vectors from the database whose metadata starts with the given prefix.

        Parameters:
        -----------
        meta_prefix : str
            The prefix of the metadata to match for deletion.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM vectors WHERE metadata LIKE ?
            ''', (f"{meta_prefix}%",))
            conn.commit()


# Example usage
if __name__ == "__main__":
    # Example with TFIDFVectorizer
    from lollmsvectordb import TFIDFVectorizer
    tfidf_vectorizer = TFIDFVectorizer()
    tfidf_vectorizer.fit(["This is a sample text.", "Another sample text.", "Another sample text 1.", "Another sample text 2.", "Another sample text 3.", "Another sample text 4."])
    db = VectorDatabase("vector_db.sqlite", tfidf_vectorizer)
    db.add_vector("This is a sample text.", meta="Vector 1")
    db.add_vector("Another sample text.", meta="Vector 2")
    db.build_index()
    results = db.search("This is a sample text.")
    print(results)
    db.load()
