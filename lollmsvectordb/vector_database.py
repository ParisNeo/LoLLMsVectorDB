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
from lollmsvectordb.tokenizer import Tokenizer
from lollmsvectordb.lollms_tokenizers.tiktoken_tokenizer import TikTokenTokenizer
from typing import List, Tuple, Optional, Union
from ascii_colors import ASCIIColors, trace_exception
import pickle
from pathlib import Path
from tqdm import tqdm

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

    def __init__(self, db_path: str, vectorizer: Vectorizer, tokenizer: Tokenizer, n_neighbors: int = 5, algorithm: str = 'auto', metrics:str="euclidean"):
        """
        Initializes the VectorDatabase with the given parameters.

        Parameters:
        -----------
        db_path : str
            Path to the SQLite database file.
        vectorizer : Vectorizer
            An instance of a vectorizer to convert data into vectors.
        tokenizer : Tokenizer
            A tokenizer to split text
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
        self.tokenizer = tokenizer
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metrics    = metrics
        self.nn_model = None
        self.vectors = []
        self.texts = []
        self.documents=[]
        self.chunks=[]
        self.titles=[]
        self.paths=[]

        if db_path!="":
            self._create_tables()
            self._load_vectors()
            self.load_first_kneighbors_model()
        self.new_data=False

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
                    title TEXT NOT NULL,
                    path TEXT NOT NULL -- Because every document needs a home, right?
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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kneighbors_model (
                    id INTEGER PRIMARY KEY,
                    model BLOB NOT NULL
                )
            ''')
            conn.commit()
            self.set_first_vectorizer_name(self.vectorizer.name)

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
        if self.db_path!="":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name FROM vectorizer_info ORDER BY id LIMIT 1')
                result = cursor.fetchone()
                return result[0] if result else ""
        else:
            return self.vectorizer.name
        
    def set_first_vectorizer_name(self, new_name: str) -> bool:
        """
        Sets the name of the first vectorizer in the database.

        Args:
            new_name (str): The new name to set for the first vectorizer.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        if self.db_path!="":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM vectorizer_info ORDER BY id LIMIT 1')
                result = cursor.fetchone()
                if result:
                    first_id = result[0]
                    cursor.execute('UPDATE vectorizer_info SET name = ? WHERE id = ?', (new_name, first_id))
                    conn.commit()
                    return True
                else:
                    cursor.execute('INSERT INTO vectorizer_info (name) VALUES (?)', (new_name,))
                return False
        else:
            return

    def add_document(self, title: str, text: str, path: Union[str, Path]="unknown", chunk_size: int = 512, force_update=False):
        """
        Adds a document and its chunks to the database.

        Parameters:
        -----------
        title : str
            The title of the document.
        text : str
            The full text of the document.
        path : Union[str, Path]
            The path to the document.
        chunk_size : int, optional
            The size of each chunk (default is 512).
        """
        doc_hash = self._hash_document(text)

        if self.db_path!="":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM documents WHERE hash = ?
                ''', (doc_hash,))
                result = cursor.fetchone()
                
                if result is not None:
                    if not force_update:
                        print(f"Document with hash {doc_hash} already exists. It's like déjà vu all over again! 🌀")
                        return
                    else:
                        cursor.execute('''
                            DELETE FROM documents WHERE hash = ?
                        ''', (doc_hash,))
                        conn.commit()
                
                cursor.execute('''
                    INSERT INTO documents (hash, title, path) VALUES (?, ?, ?)
                ''', (doc_hash, title, str(path)))
                document_id = cursor.lastrowid

                paragraphs = text.split('\n')
                current_chunk = ""
                current_tokens = []
                ASCIIColors.multicolor(["Chunking file:",f"{title}"],[ASCIIColors.color_red, ASCIIColors.color_yellow])
                for paragraph in tqdm(paragraphs):
                    paragraph_tokens = self.tokenizer.tokenize(paragraph)
                    if len(current_tokens) + len(paragraph_tokens) > chunk_size:
                        vector = self.vectorizer.vectorize([current_chunk])[0]
                        vector_blob = np.array(vector).tobytes()
                        cursor.execute('''
                            INSERT INTO chunks (document_id, vector, text, title) VALUES (?, ?, ?, ?)
                        ''', (document_id, vector_blob, current_chunk, title))
                        current_chunk = paragraph
                        current_tokens = paragraph_tokens
                    else:
                        current_chunk += "\n" + paragraph
                        current_tokens.extend(paragraph_tokens)

                if current_chunk:
                    vector = self.vectorizer.vectorize([current_chunk])[0]
                    vector_blob = np.array(vector).tobytes()
                    cursor.execute('''
                        INSERT INTO chunks (document_id, vector, text, title) VALUES (?, ?, ?, ?)
                    ''', (document_id, vector_blob, current_chunk, title))

                conn.commit()
                self.new_data=True
        else:
            paragraphs = text.split('\n')
            current_chunk = ""
            current_tokens = []
            self.documents.append({"title":title,"hash":doc_hash})
            self.titles.append(title)
            self.paths.append(path)
            ASCIIColors.multicolor(["Chunking file:",f"{title}"],[ASCIIColors.color_red, ASCIIColors.color_yellow])
            for i,paragraph in enumerate(tqdm(paragraphs)):
                paragraph_tokens = self.tokenizer.tokenize(paragraph)
                if len(current_tokens) + len(paragraph_tokens) > chunk_size or i==len(paragraphs)-1:
                    vector = self.vectorizer.vectorize([current_chunk])[0]
                    vector_blob = np.array(vector).tobytes()
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
                    self.chunks.append({
                        "title":title,
                        "chunk":current_chunk,
                        "vector":vector
                    })
                    self.texts.append(current_chunk)
                else:
                    current_chunk += "\n" + paragraph
                    current_tokens.extend(paragraph_tokens)



    def remove_document(self, doc_hash: str):
        """
        Removes a document and its chunks from the database.

        Parameters:
        -----------
        doc_hash : str
            The hash of the document to be removed.
        """
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Delete chunks associated with the document
                cursor.execute('''
                    DELETE FROM chunks WHERE document_id IN (
                        SELECT id FROM documents WHERE hash = ?
                    )
                ''', (doc_hash,))
                # Delete the document itself
                cursor.execute('''
                    DELETE FROM documents WHERE hash = ?
                ''', (doc_hash,))
                conn.commit()
        else:
            # Remove document from in-memory storage
            try:
                index = [i for i,_ in enumerate(self.chunks) if [doc for doc in self.documents if doc["hash"]==doc_hash][0] == doc_hash]
                doc_id = self.documents.index(self.chunks[index]["title"])
                del self.documents[doc_id]
                del self.titles[doc_id]
                del self.paths[doc_id]
                del self.texts[index]
                del self.chunks[index]
                print(f"Document with hash '{doc_hash}' removed from in-memory storage.")
            except:
                ASCIIColors.error("Document Not found!")

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
        if self.db_path!="":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 1 FROM documents WHERE hash = ?
                ''', (doc_hash,))
                return cursor.fetchone() is not None
        else:
            return len([d for d in self.documents if d["hash"]==doc_hash])>0
        
    def _load_vectors(self):
        """
        Loads vectors and their text from the database into memory.
        """
        if self.db_path!="":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT chunks.vector, chunks.text, documents.title, documents.path 
                    FROM chunks 
                    JOIN documents ON chunks.document_id = documents.id
                ''')
                rows = cursor.fetchall()
                self.vectors = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
                self.texts = [row[1] for row in rows]
                self.titles = [row[2] for row in rows]
                self.paths = [row[3] for row in rows]
        else:
            ASCIIColors.error("Can't load vectors from database if you don't specify a file path")
    def _update_vectors(self):
        """
        Updates vectors in the database using the current vectorizer.
        """
        if self.db_path!="":
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
        else:
            try:
                for chunk in self.chunks:
                    text = chunk["chunk"]
                    vector = self.vectorizer.vectorize([text])[0]
                    chunk["vector"] = vector
                    self.vectors.append(vector)
            except Exception as ex:
                trace_exception(ex)
                ASCIIColors.error("Document Not found!")


    def store_kneighbors_model(self) -> None:
        """
        Store the KNeighbors model into the SQLite database as the only entry.
        """
        if self.db_path=="":
            return
        # Serialize the model using pickle
        model_blob = pickle.dumps(self.n_neighbors)
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear out any existing entries
        cursor.execute('DELETE FROM kneighbors_model')
        
        # Insert the model into the database
        cursor.execute('''
            INSERT INTO kneighbors_model (model)
            VALUES (?)
        ''', (model_blob,))
        
        # Commit the transaction and close the connection
        conn.commit()
        conn.close()
    
    def load_first_kneighbors_model(self):
        """
        Load the first KNeighbors model from the SQLite database.
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Retrieve the first model from the database
        cursor.execute('''
            SELECT model FROM kneighbors_model ORDER BY id LIMIT 1
        ''')
        model_blob = cursor.fetchone()
        
        # Close the connection
        conn.close()
        
        if model_blob is None:
            ASCIIColors.yellow("No model found in the database")
            self.build_index()
            return
        
        # Deserialize the model using pickle
        self.n_neighbors = pickle.loads(model_blob[0])
        

    def build_index(self, force_reindex= False):
        """
        Builds the nearest neighbors index using the loaded vectors.
        """
        if self.db_path=="":
            self._update_vectors()
            self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric='cosine')
            if len(self.vectors)>0:
                self.nn_model.fit(self.vectors)
                self.store_kneighbors_model()
            return
        
        if self.vectorizer.name != self.get_first_vectorizer_name() or force_reindex:
            if self.vectorizer.name != self.get_first_vectorizer_name() and self.get_first_vectorizer_name()!="":
                ASCIIColors.warning("Detected a change in the vectorizer. Revectorizing the whole database")
            self._update_vectors()
            self.set_first_vectorizer_name(self.vectorizer.name)
        else:
            self._load_vectors()
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric='cosine')
        if len(self.vectors)>0:
            self.nn_model.fit(self.vectors)
            self.store_kneighbors_model()

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
            A list of tuples containing the vector, text, title, path, and distance of the nearest neighbors.
        """
        if self.nn_model is None:
            raise ValueError("Index not built. Call build_index() before searching.")
        
        if len(self.vectors) < n_results:
            n_results=len(self.vectors)
        
        query_vector = self.vectorizer.vectorize([query_data])[0]
        distances, indices = self.nn_model.kneighbors([query_vector], n_neighbors=n_results)
        results = [(self.vectors[idx], self.texts[idx], self.titles[idx], self.paths[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
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
    from lollmsvectordb.lollms_vectorizers.bert_vectorizer import BERTVectorizer
    tfidf_vectorizer = BERTVectorizer()
    tfidf_vectorizer.fit(["This is a sample text.", "Another sample text.", "Another sample text 1.", "Another sample text 2.", "Another sample text 3.", "Another sample text 4."])
    tokenizer = TikTokenTokenizer()
    db = VectorDatabase("vector_db.sqlite", tfidf_vectorizer, tokenizer) # 

    # Add multiple documents to the database
    documents = [
        ("Document 1", "This is the first sample text."),
        ("Document 2", "Here is another example of a sample text."),
        ("Document 3", "This document is different from the others."),
        ("Document 4", "Yet another document with some sample text."),
        ("Document 5", "This is the fifth document in the database."),
        ("Document 6", "Finally, this is the sixth sample text. welcome to the moon")
    ]

    for title, text in documents:
        db.add_document(title, text)

    # Build the nearest neighbors index
    db.build_index()

    # Perform a search query
    query = "what is the sixth sample text."
    results = db.search(query, n_results=3)

    # Print the search results
    for vector, text, title, distance in results:
        print(f"Title: {title}, Text: {text}, Distance: {distance}")
