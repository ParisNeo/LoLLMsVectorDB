"""
LoLLMsVectorDB

File: bert_vectorizer.py
Author: ParisNeo
Description: Contains the BERTVectorizer class for vectorizing text data using BERT.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from lollmsvectordb.vectorizer import Vectorizer
from ascii_colors import ASCIIColors
from typing import List
import json

class BERTVectorizer(Vectorizer):
    _model_cache = {}
    _model_ref_count = {}
    def __init__(self, model_name: str = 'bert-base-nli-mean-tokens'):
        """
        Initializes the BERTVectorizer with a specified BERT model.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
        """
        super().__init__("BertVectorizer")
        # ASCIIColors.multicolor(["LollmsVectorDB>","Loading Pretrained Bert Tokenizer ..."],[ASCIIColors.color_red, ASCIIColors.color_cyan], end="", flush=True)
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        ASCIIColors.success("OK")
        ASCIIColors.multicolor(["LollmsVectorDB>",f"Loading Pretrained Bert model {model_name} ..."],[ASCIIColors.color_red, ASCIIColors.color_cyan], end="", flush=True)

        if model_name in BERTVectorizer._model_cache:
            self.model_ = BERTVectorizer._model_cache[model_name]
            BERTVectorizer._model_ref_count[model_name] += 1
            ASCIIColors.success(f"Loaded {model_name} from cache")
        else:
            ASCIIColors.multicolor(["LollmsVectorDB>", f"Loading Pretrained Bert model {model_name} ..."], [ASCIIColors.color_red, ASCIIColors.color_cyan], end="", flush=True)
            self.model_ = SentenceTransformer(model_name)
            BERTVectorizer._model_cache[model_name] = self.model_
            BERTVectorizer._model_ref_count[model_name] = 1
            ASCIIColors.success("OK")

        ASCIIColors.success("OK")
        self.parameters = {
            "model_name": model_name
        }

        ASCIIColors.multicolor(["LollmsVectorDB>",f" Parameters:"],[ASCIIColors.color_red, ASCIIColors.color_bright_green])
        ASCIIColors.yellow(json.dumps(self.parameters, indent=4))
        self.fitted = True

    def __del__(self):
        """
        Destructor to manage the lifecycle of the model.
        """
        if self.parameters["model_name"] in BERTVectorizer._model_ref_count:
            BERTVectorizer._model_ref_count[self.parameters["model_name"]] -= 1
            if BERTVectorizer._model_ref_count[self.parameters["model_name"]] == 0:
                del BERTVectorizer._model_cache[self.parameters["model_name"]]
                del BERTVectorizer._model_ref_count[self.parameters["model_name"]]
                ASCIIColors.success(f"Released model {self.parameters['model_name']} from cache")

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
        sentence_embeddings  = self.model_.encode(data)
        return sentence_embeddings

    def get_models(self):
        """
        Returns a list of model names
        """
        return [
        "bert-base-uncased",
        "bert-base-multilingual-uncased",
        "bert-large-uncased",
        "bert-large-uncased-whole-word-masking-finetuned-squad",
        ]
    def __str__(self):
        model_name = self.parameters['model_name']
        return f'Lollms Vector DB BERTVectorizer. Using model {model_name}.'

    def __repr__(self):
        model_name = self.parameters['model_name']
        return f'Lollms Vector DB BERTVectorizer. Using model {model_name}.'
