"""
LoLLMsVectorDB

File: tfidf_vectorizer.py
Author: ParisNeo
Description: Contains the TFIDFVectorizer class for vectorizing text data.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from lollmsvectordb.vectorizer import Vectorizer
from typing import List

class TFIDFVectorizer(Vectorizer):
    def __init__(self):
        """
        Initialize the TFIDFVectorizer with the name and the Sklearn TfidfVectorizer.
        """
        super().__init__("TFIDFVectorizer", True)
        self.model = None
        self.fitted = False

    def fit(self, data: List[str]) -> None:
        """
        Fit the TFIDF vectorizer to the data.

        Args:
            data (List[str]): The data to fit the vectorizer on.
        """
        self.model = SklearnTfidfVectorizer()
        self.model.fit(data)
        self.fitted = True

    def vectorize(self, data: List[str]) -> List[List[float]]:
        """
        Transform the data into TFIDF vectors.

        Args:
            data (List[str]): The data to transform.

        Returns:
            List[List[float]]: The transformed data as TFIDF vectors.
        """
        return self.model.transform(data).toarray()
    
    def __str__(self):
        return f'Lollms Vector DB TFIDFVectorizer.'

    def __repr__(self):
        return f'Lollms Vector DB TFIDFVectorizer.'
