"""
LoLLMsVectorDB

File: bert_vectorizer.py
Author: ParisNeo
Description: Contains the BERTVectorizer class for vectorizing text data using Hugging Face's transformers.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""
from ascii_colors import ASCIIColors, trace_exception
try:
    from transformers import AutoTokenizer, AutoModel
except Exception as ex:
    trace_exception(ex)
    ASCIIColors.red("transformers has a problem. Reinstalling it")
    import pipmaster as pm
    pm.install("transformers", force_reinstall=True, upgrade=True)
    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as ex:
        trace_exception(ex)
        ASCIIColors.error("Warning! Transformers is broken. Try to manually reinstall pytorch and then reinstall transformers")
        class AutoTokenizer:
            def __init__(self, model_name="") -> None:
                self.model = model_name
        class AutoModel:
            def __init__(self, model_name="") -> None:
                self.model = model_name

import torch
import numpy as np
from lollmsvectordb.vectorizer import Vectorizer
from typing import List
import json

class BERTVectorizer(Vectorizer):
    _model_cache = {}
    _model_ref_count = {}

    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initializes the BERTVectorizer with a specified BERT model.

        Args:
            model_name (str): The name of the pre-trained BERT model to use.
        """
        super().__init__("BertVectorizer")
        ASCIIColors.multicolor(["LollmsVectorDB>", f"Loading Pretrained BERT model {model_name} ..."], [ASCIIColors.color_red, ASCIIColors.color_cyan], end="", flush=True)

        if model_name in BERTVectorizer._model_cache:
            self.tokenizer, self.model_ = BERTVectorizer._model_cache[model_name]
            BERTVectorizer._model_ref_count[model_name] += 1
            ASCIIColors.success(f"Loaded {model_name} from cache")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_ = AutoModel.from_pretrained(model_name)
            BERTVectorizer._model_cache[model_name] = (self.tokenizer, self.model_)
            BERTVectorizer._model_ref_count[model_name] = 1
            ASCIIColors.success("OK")

        self.parameters = {
            "model_name": model_name
        }

        ASCIIColors.multicolor(["LollmsVectorDB>", f" Parameters:"], [ASCIIColors.color_red, ASCIIColors.color_bright_green])
        ASCIIColors.yellow(json.dumps(self.parameters, indent=4))
        self.fitted = True

    def __del__(self):
        """
        Destructor to manage the lifecycle of the model.
        """
        try:
            if self.parameters["model_name"] in BERTVectorizer._model_ref_count:
                BERTVectorizer._model_ref_count[self.parameters["model_name"]] -= 1
                if BERTVectorizer._model_ref_count[self.parameters["model_name"]] == 0:
                    del BERTVectorizer._model_cache[self.parameters["model_name"]]
                    del BERTVectorizer._model_ref_count[self.parameters["model_name"]]
                    ASCIIColors.success(f"Released model {self.parameters['model_name']} from cache")
        except Exception as ex:
            trace_exception(ex)

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
            with torch.no_grad():
                outputs = self.model_(**inputs)
            # Use the mean of the last hidden state as the sentence embedding
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(sentence_embedding)
        return embeddings

    def get_models(self):
        """
        Returns a list of model names
        """
        return [
            "bert-base-uncased",
            "bert-base-multilingual-uncased",
            "bert-large-uncased",
            "bert-large-uncased-whole-word-masking-finetuned-squad",
            "distilbert-base-uncased",
            "roberta-base",
            "roberta-large",
            "xlm-roberta-base",
            "xlm-roberta-large",
            "albert-base-v2",
            "albert-large-v2",
            "albert-xlarge-v2",
            "albert-xxlarge-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-distilroberta-v1",
            "sentence-transformers/all-mpnet-base-v2"
        ]

    def __str__(self):
        model_name = self.parameters['model_name']
        return f'Lollms Vector DB BERTVectorizer. Using model {model_name}.'

    def __repr__(self):
        model_name = self.parameters['model_name']
        return f'Lollms Vector DB BERTVectorizer. Using model {model_name}.'
