"""
LoLLMsVectorDB

File: bert_vectorizer.py
Author: ParisNeo
Description: Contains the BERTVectorizer class for vectorizing text data using BERT.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from lollmsvectordb.vectorizer import Vectorizer
from typing import List

class BERTVectorizer(Vectorizer):
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initializes the BERTVectorizer with a specified BERT model.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
        """
        super().__init__("BertVectorizer")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def fit(self, data: List[str]):
        """
        BERT does not require fitting as it is a pre-trained model.
        This method is included to maintain consistency with the Vectorizer interface.

        Args:
            data (List[str]): The data to fit on (not used).
        """
        pass

    def vectorize(self, data: List[str]) -> List[np.ndarray]:
        """
        Vectorizes the input data using BERT.

        Args:
            data (List[str]): The data to vectorize.

        Returns:
            List[np.ndarray]: The list of BERT embeddings for each input text.
        """
        embeddings = []
        for text in data:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            # Use the mean of the token embeddings as the sentence embedding
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
            # Flatten the tensor to 1D
            flattened_embedding = sentence_embedding.detach().numpy().flatten()
            embeddings.append(flattened_embedding)
        return embeddings
