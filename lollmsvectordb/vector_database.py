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
import json
from pathlib import Path
from tqdm import tqdm

from lollmsvectordb.database_elements.document import Document
from lollmsvectordb.database_elements.chunk import Chunk

from lollmsvectordb.text_chunker import TextChunker
from lollmsvectordb.llm_model import LLMModel

__version__ = 3

def replace_nan_with_zero(arrays: List[np.ndarray]) -> List[int]:
    """
    Replace NaN values with 0 in the arrays and return indices of arrays that contained NaN values.

    Args:
        arrays (List[np.ndarray]): List of NumPy arrays.

    Returns:
        List[int]: List of indices of arrays that contained NaN values.
    """
    nan_indices = []
    for i, array in enumerate(arrays):
        if np.isnan(array).any():
            arrays[i] = np.nan_to_num(array, nan=0.0)
            nan_indices.append(i)
    return nan_indices

def find_nan_indices(arrays: List[np.ndarray]) -> List[int]:
    """
    Find indices of arrays that contain NaN values.

    Args:
        arrays (List[np.ndarray]): List of NumPy arrays.

    Returns:
        List[int]: List of indices of arrays that contain NaN values.
    """
    nan_indices = [i for i, array in enumerate(arrays) if np.isnan(array).any()]
    return nan_indices

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

    def __init__(self, db_path: str, vectorizer: Vectorizer, tokenizer: Tokenizer, chunk_size: int = 512, overlap: int = 0, clean_chunks=True, n_neighbors: int = 5, algorithm: str = 'auto', metrics:str="euclidean", reset=False, model: Optional[LLMModel] = None):
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
        chunk_size : int, optional
            The maximum size of each chunk in tokens (default is 512).
        clean_chunks : bool, optional
            If true, then the chunks will be cleaned by removing all extra line returns (default is True).
            
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
        reset : bool, optional
            If True, this means that any existing data will be removed from the database and replaced with a new vector database. (default is False)
        """
        self.db_path = db_path
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer
        self.n_neighbors = n_neighbors
        self.chunk_size = chunk_size
        self.algorithm = algorithm
        self.metrics    = metrics
        self.clean_chunks = clean_chunks
        self.nn_model = None
        self.textChunker = TextChunker(chunk_size=chunk_size, overlap=overlap, model=model)
        self.documents:List[Document]=[]
        self.chunks:List[Chunk]=[]
        self.nn_fitted = False

        if db_path!="":
            self._create_tables(reset=reset)
            self._load_vectors()
            try:
                self.load_vectorizer_model()
            except Exception as ex:
                pass
            try:
                self.load_first_kneighbors_model()
            except Exception as ex:
                if len(self.vectors)>0:
                    indices = find_nan_indices(self.vectors)
                    self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric='cosine')
                    self.nn_model.fit(self.vectors)
                    self.nn_fitted = True
                    self.store_kneighbors_model()
        self.build_index(False)
        ASCIIColors.multicolor(["lollmsVectorDB>","Vectorizer status:",f"{self.vectorizer}"],[ASCIIColors.color_red,ASCIIColors.color_cyan, ASCIIColors.color_yellow])
        ASCIIColors.multicolor(["lollmsVectorDB>","Search model status:",f"{self.nn_model}"],[ASCIIColors.color_red,ASCIIColors.color_cyan, ASCIIColors.color_yellow])
        ASCIIColors.multicolor(["lollmsVectorDB>","lollmsVectorDB ",f"is ready"],[ASCIIColors.color_red,ASCIIColors.color_cyan, ASCIIColors.color_yellow])
        self.new_data=False


    def _create_tables(self, reset: bool = False):
        """
        Creates the necessary tables in the SQLite database if they do not exist.
        If reset is True, it will drop the existing tables and recreate them.

        Args:
            reset (bool): If True, drops existing tables and recreates them.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if reset:
                cursor.execute('DROP TABLE IF EXISTS documents')
                cursor.execute('DROP TABLE IF EXISTS chunks')
                cursor.execute('DROP TABLE IF EXISTS vectorizer_info')
                cursor.execute('DROP TABLE IF EXISTS kneighbors_model')
                cursor.execute('DROP TABLE IF EXISTS database_info')

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
                    vector BLOB,
                    text TEXT NOT NULL,
                    nb_tokens INT NOT NULL,
                    chunk_id INT, 
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vectorizer_info (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    parameters TEXT,
                    model BLOB
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kneighbors_model (
                    id INTEGER PRIMARY KEY,
                    model BLOB NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS database_info (
                    id INTEGER PRIMARY KEY,
                    version INT NOT NULL
                )
            ''')
            # Check if there is an entry in the database_info table
            cursor.execute('SELECT COUNT(*) FROM database_info')
            count = cursor.fetchone()[0]

            # If there is no entry, insert the version number
            if count == 0:
                cursor.execute('INSERT INTO database_info (version) VALUES (?)', (__version__,))
                conn.commit()            
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


    def get_version(self) -> Optional[int]:
        """
        Retrieve the version of the database from the database_info table.

        Returns:
            Optional[int]: The version of the database if it exists, otherwise None.
        """
        if self.db_path:
            db_file = Path(self.db_path)
            if db_file.exists():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS database_info (
                            id INTEGER PRIMARY KEY,
                            version INT NOT NULL
                        )
                    ''')
                    cursor.execute('SELECT version FROM database_info WHERE id = 1')
                    result = cursor.fetchone()
                    if result:
                        return result[0]
        return None




    def add_document(self, title: str, text: str, path: Union[str, Path]="unknown", force_update=False, min_nb_tokens_in_chunk=10):
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
                        print(f"Document with hash {doc_hash} already exists")
                        return
                    else:
                        cursor.execute('''
                            DELETE FROM documents WHERE hash = ?
                        ''', (doc_hash,))
                        conn.commit()

                doc = Document(doc_hash, title, path, len(self.documents))
                self.documents.append(doc)
                cursor.execute('''
                    INSERT INTO documents (hash, title, path) VALUES (?, ?, ?)
                ''', (doc_hash, title, str(path)))
                document_id = cursor.lastrowid

                ASCIIColors.multicolor(["lollmsVectorDB> ","Chunking file:",f"{title}"],[ASCIIColors.color_red,ASCIIColors.color_cyan, ASCIIColors.color_yellow])
                if self.textChunker.model:
                    ASCIIColors.multicolor(["lollmsVectorDB> ","Preprocessing chunks is active"],[ASCIIColors.color_red,ASCIIColors.color_cyan, ASCIIColors.color_yellow])

                chunks:List[Chunk]= self.textChunker.get_text_chunks(text, doc, min_nb_tokens_in_chunk=min_nb_tokens_in_chunk)

                for chunk in tqdm(chunks):
                    if not self.vectorizer.requires_fitting or self.vectorizer.model is not None:
                        vector = self.vectorizer.vectorize([chunk.text])[0]
                        vector_blob = np.array(vector).tobytes()
                        cursor.execute('''
                            INSERT INTO chunks (document_id, vector, text, nb_tokens, chunk_id) VALUES (?, ?, ?, ?, ?)
                        ''', (document_id, vector_blob, chunk.text, chunk.nb_tokens, chunk.chunk_id))
                    else:
                        cursor.execute('''
                            INSERT INTO chunks (document_id, text, nb_tokens) VALUES (?, ?, ?)
                        ''', (document_id, chunk.text, chunk.nb_tokens))
                conn.commit()
                self.new_data=True
        else:
            chunks = self.textChunker.get_text_chunks(text, doc, self.clean_chunks)
            doc = Document(doc_hash, title, path, len(self.documents))
            self.documents.append(doc)
            for chunk in chunks:
                chunk.vector = self.vectorizer.vectorize([chunk.text])[0]


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
                doc = [d for d in self.documents if d.hash==doc_hash][0]
                self.documents = [d for d in self.documents if d.hash!=doc_hash]
                self.chunks = [c for c in self.chunks if c.doc!=doc]
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
            return len([d for d in self.documents if d.hash==doc_hash])>0
        
    def _load_vectors(self):
        """
        Loads vectors and their text from the database into memory.
        """
        if self.db_path!="":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT chunks.vector
                    FROM chunks 
                ''')
                rows = cursor.fetchall()
                self.vectors = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
        else:
            ASCIIColors.error("Can't load vectors from database if you don't specify a file path")

    def _update_vectors(self, revectorize=False):
        """
        Updates vectors in the database using the current vectorizer.
        """
        if self.db_path!="":
            self.vectors = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, text, vector FROM chunks')
                rows = cursor.fetchall()
                ASCIIColors.multicolor(["LollmsVectorDB> ", f"Vectorizing {len(rows)} chunks"], [ASCIIColors.color_red, ASCIIColors.color_cyan])
                for row in tqdm(rows):
                    chunk_id, text, vector = row
                    if vector is None or revectorize:
                        vector = np.array(self.vectorizer.vectorize([text])[0])
                        self.vectors.append(vector)
                        vector_blob = vector.tobytes()
                        cursor.execute('UPDATE chunks SET vector = ? WHERE id = ?', (vector_blob, chunk_id))
                    else:
                        self.vectors.append(np.frombuffer(vector, dtype=np.float32))
                conn.commit()
        else:
            self.vectors = []
            try:
                ASCIIColors.multicolor(["LollmsVectorDB> ", f"Vectorizing {len(self.chunks)} chunks"], [ASCIIColors.color_red, ASCIIColors.color_cyan])
                for chunk in tqdm(self.chunks):
                    vector = self.vectorizer.vectorize([chunk.text])[0]
                    chunk.vector = vector
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
        model_blob = pickle.dumps(self.nn_model)
        
        # Connect to the database
        with sqlite3.connect(self.db_path) as conn:
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

    
    def load_first_kneighbors_model(self):
        """
        Load the first KNeighbors model from the SQLite database.
        """
        # Connect to the database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Retrieve the first model from the database
            cursor.execute('''
                SELECT model FROM kneighbors_model ORDER BY id LIMIT 1
            ''')
            model_blob = cursor.fetchone()
                        
            if model_blob is None:
                ASCIIColors.yellow("No nneighbors model found in the database")
                return
            
            # Deserialize the model using pickle
            self.nn_model = pickle.loads(model_blob[0])
            self.nn_fitted = True
        
    def load_vectorizer_model(self, force_new_vectorizer=True) -> Optional[str]:
        ASCIIColors.multicolor(["LollmsVectorDB> ", "Loading vectorizer"], [ASCIIColors.color_red, ASCIIColors.color_cyan])
        if self.db_path!="":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, model, parameters FROM vectorizer_info ORDER BY id LIMIT 1')
                result = cursor.fetchone()
                if not result:
                    self.store_vectorizer_model()
                    return
                if self.vectorizer.name != result[0]:
                    if force_new_vectorizer:
                        return
                    else:
                        if result[0]=="BertVectorizer":
                            params = json.loads(result[2])
                            self.vectorizer = BERTVectorizer(params["model_name"])
                        elif  result[0]=="TFIDFVectorizer":
                            self.vectorizer = TFIDFVectorizer()
                else:
                    if self.vectorizer.requires_fitting and result[1]:
                        self.vectorizer.model = pickle.loads(result[1])
                        self.vectorizer.fitted = True
                return (result[0], result[1], result[2]) if result else ("", None, None)
        else:
            return self.vectorizer.name, self.vectorizer.model, self.vectorizer.parameters
        
    def store_vectorizer_model(self) -> bool:
        """
        Sets the vectorizer data in the database.

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
                    cursor.execute('UPDATE vectorizer_info SET name = ? WHERE id = ?', (self.vectorizer.name, first_id))
                    if self.vectorizer.model:
                        model_blob = pickle.dumps(self.vectorizer.model)
                        cursor.execute('UPDATE vectorizer_info SET model = ? WHERE id = ?', (model_blob, first_id))
                    if self.vectorizer.parameters:
                        vectorizer_parameters = json.dumps(self.vectorizer.parameters)
                        cursor.execute('UPDATE vectorizer_info SET parameters = ? WHERE id = ?', (vectorizer_parameters, first_id))
                    conn.commit()
                    return True
                else:
                    cursor.execute('INSERT INTO vectorizer_info (name) VALUES (?)', (self.vectorizer.name,))
                return False
        else:
            return



    def build_index(self, revectorize:bool=True):
        """
        Builds the nearest neighbors index using the loaded vectors.
        """
        ASCIIColors.multicolor(["LollmsVectorDB> ", "Indexing database"], [ASCIIColors.color_red, ASCIIColors.color_cyan])
        self.load_vectorizer_model()
        if self.vectorizer.fitted:
            ASCIIColors.multicolor(["LollmsVectorDB> ", "Vectorizer is ready"], [ASCIIColors.color_red, ASCIIColors.color_green])

        if self.vectorizer.requires_fitting and self.vectorizer.model is None:
            if self.db_path!="":
                ASCIIColors.multicolor(["LollmsVectorDB> ", "Fitting vectorizer"], [ASCIIColors.color_red, ASCIIColors.color_cyan])
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT text FROM chunks
                    ''')
                    chunks = cursor.fetchall()
                    if len(chunks)==0:
                        return 
                    ASCIIColors.multicolor(["LollmsVectorDB> ", "Training vectorizer"], [ASCIIColors.color_red, ASCIIColors.color_cyan])
                    try:
                        self.vectorizer.fit([c[0] for c in chunks])
                    except:
                        self.vectorizer.model = None
                    self.store_vectorizer_model()
                    self._update_vectors(revectorize)
            else:
                self.vectorizer.fit([c.text for c in self.chunks])
                self.store_vectorizer_model()
                self._update_vectors(revectorize)
        else:
            self._load_vectors()
        
        if len(self.vectors)>0 and not self.nn_model:
            self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm, metric='cosine')
            self.nn_model.fit(self.vectors)
            self.nn_fitted = True
            self.store_kneighbors_model()
        

    def find_document_by_path(self, target_path: str) -> Optional[Document]:
        """
        Finds a document in the list of documents by its path.

        Args:
            documents (List[Document]): List of Document objects.
            target_path (str): The path to search for.

        Returns:
            Optional[Document]: The Document object with the matching path, or None if not found.
        """
        target_path = Path(target_path)
        for document in self.documents:
            if document.path == target_path:
                return document
        return None

    def search(self, query_data: str, n_results: int = 5, exclude_chunk_ids: List[int] = []) -> List[Chunk]:
        """
        Searches for the nearest neighbors of the query data.

        Parameters:
        -----------
        query_data : str
            The data to be vectorized and searched in the database.
        n_results : int, optional
            Number of nearest neighbors to return (default is 5).
        exclude_chunk_ids : List[int], optional
            List of chunk IDs to exclude from the search results (default is empty list).

        Returns:
        --------
        list of tuples
            A list of tuples containing the vector, text, title, path, and distance of the nearest neighbors.
        """
        if self.nn_model is None:
            raise ValueError("Index not built. Call build_index() before searching.")
        
        results: List[Chunk] = []
        if len(self.vectors) < n_results:
            n_results = len(self.vectors)
        
        query_vector = self.vectorizer.vectorize([query_data])[0]
        distances, indices = self.nn_model.kneighbors([query_vector], n_neighbors=n_results)
        if self.db_path != "":
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for index, distance in zip(indices[0, :], distances[0, :]):
                    # SQL query to join documents and chunks tables and retrieve the required details
                    query = '''
                        SELECT d.title, d.path, d.hash, c.text, c.nb_tokens, c.chunk_id
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.vector = ? AND c.chunk_id NOT IN ({})
                    '''.format(','.join('?' for _ in exclude_chunk_ids))

                    # Execute the query with the provided vector and exclude_chunk_ids
                    cursor.execute(query, (self.vectors[index], *exclude_chunk_ids))
                    result = cursor.fetchone()
                    if result:
                        doc = self.find_document_by_path(result[1])
                        if not doc:
                            doc = Document(result[2], result[0], result[1], len(self.documents))
                        chunk = Chunk(doc, self.vectors[index], result[3], result[4], distance=distance, chunk_id=result[5])
                        results.append(chunk)
        else:
            results = [c for c in self.chunks if c.vector in self.vectors[indices] and c.chunk_id not in exclude_chunk_ids]

        return results


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

    
    db = VectorDatabase("vector_db.sqlite", BERTVectorizer(), TikTokenTokenizer(),chunk_size=512, clean_chunks=True) # 

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
    results:List[Chunk] = db.search(query, n_results=3)

    # Print the search results
    for chunk in results:
        print(f"Title: {chunk.doc.title}, Text: {chunk.text}, Distance: {chunk.distance}, NB tokens: {chunk.nb_tokens}")
