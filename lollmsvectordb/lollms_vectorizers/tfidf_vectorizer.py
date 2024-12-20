"""
LoLLMsVectorDB

File: tfidf_vectorizer.py
Author: ParisNeo
Description: Contains the TFIDFVectorizer class for vectorizing text data.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

import math
from typing import List

import numpy as np

from lollmsvectordb.vectorizer import Vectorizer


class TFIDFVectorizer(Vectorizer):
    def __init__(self):
        """
        Initialize the TFIDFVectorizer with the name.
        """
        super().__init__("TFIDFVectorizer", True)
        self.parameters = {"model_name": ""}
        self.vocab = {}
        self.idf = {}
        self.fitted = False

    def fit(self, data: List[str]) -> None:
        """
        Fit the TFIDF vectorizer to the data.

        Args:
            data (List[str]): The data to fit the vectorizer on.
        """
        # Build vocabulary
        vocab_set = set()
        for document in data:
            vocab_set.update(document.split())
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}

        # Calculate IDF
        doc_count = len(data)
        word_doc_count = np.zeros(len(self.vocab))
        for document in data:
            unique_words = set(document.split())
            for word in unique_words:
                if word in self.vocab:
                    word_doc_count[self.vocab[word]] += 1

        self.idf = np.log(doc_count / (1 + word_doc_count))
        self.fitted = True

    def _compute_tfidf(
        self, tf: np.ndarray, idf: np.ndarray, doc_len: int
    ) -> np.ndarray:
        """
        Compute the TF-IDF vector for a single document.

        Args:
            tf (np.ndarray): Term frequency of words in the document.
            idf (np.ndarray): IDF values for the vocabulary.
            doc_len (int): Length of the document.

        Returns:
            np.ndarray: The TF-IDF vector for the document.
        """
        return (tf / doc_len) * idf

    def vectorize(self, data: List[str]) -> List[List[float]]:
        """
        Transform the data into TFIDF vectors.

        Args:
            data (List[str]): The data to transform.

        Returns:
            List[List[float]]: The transformed data as TFIDF vectors.
        """
        if not self.fitted:
            self.fit(data)

        vectors = []
        for document in data:
            tf = np.zeros(len(self.vocab))
            words = document.split()
            for word in words:
                if word in self.vocab:
                    tf[self.vocab[word]] += 1
            doc_len = len(words)
            doc_vector = self._compute_tfidf(tf, self.idf, doc_len)
            vectors.append(doc_vector)

        return vectors

    def __str__(self):
        return f"Lollms Vector DB TFIDFVectorizer."

    def __repr__(self):
        return f"Lollms Vector DB TFIDFVectorizer."
