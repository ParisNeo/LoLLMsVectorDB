"""
LoLLMsVectorDB

File: tfidf_vectorizer.py
Author: ParisNeo
Description: Contains the TFIDFVectorizer class for vectorizing text data.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from lollmsvectordb.vectorizer import Vectorizer

class TFIDFVectorizer(Vectorizer):
    def __init__(self):
        super().__init__("TFIDFVectorizer")
        self.vectorizer = SklearnTfidfVectorizer()

    def fit(self, data):
        self.vectorizer.fit(data)

    def vectorize(self, data):
        return self.vectorizer.transform(data).toarray()
