"""
LoLLMsVectorDB

File: word2vec_vectorizer.py
Author: ParisNeo
Description: Contains the Word2VecVectorizer class for vectorizing text data.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

from gensim.models import Word2Vec
from lollmsvectordb.vectorizer import Vectorizer

class Word2VecVectorizer(Vectorizer):
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.model = Word2Vec(size=size, window=window, min_count=min_count, workers=workers)

    def fit(self, data):
        self.model.build_vocab(data)
        self.model.train(data, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def vectorize(self, data):
        return [self.model.wv[word] for word in data if word in self.model.wv]
