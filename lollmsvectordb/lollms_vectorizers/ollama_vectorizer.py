"""
LoLLMsVectorDB

File: ollama_vectorizer.py
Author: Lambda
Description: Contains the OllamaVectorizer class for vectorizing text data using Ollama's embedding service.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""

import json
from typing import List

import numpy as np
import requests
from ascii_colors import ASCIIColors, trace_exception

from lollmsvectordb.vectorizer import Vectorizer


class OllamaVectorizer(Vectorizer):
    def __init__(self, model_name: str = "bge-m3", url: str = "http://localhost:11434"):
        """
        Initializes the OllamaVectorizer with a specified Ollama model and server details.

        Args:
            model_name (str): The name of the Ollama model to use for embeddings. Default is 'bge-m3'.
            url (str): The host url of the Ollama server.
        """
        super().__init__("OllamaVectorizer")
        self.model_name = model_name
        self.base_url = f"{url}/api/embed"

        self.parameters = {
            "model_name": self.model_name,
        }
        ASCIIColors.multicolor(
            ["LollmsVectorDB>", f"Using Ollama model {model_name} for embeddings."],
            [ASCIIColors.color_red, ASCIIColors.color_cyan],
            end="",
            flush=True,
        )
        ASCIIColors.success("OK")
        ASCIIColors.multicolor(
            ["LollmsVectorDB>", f" Parameters:"],
            [ASCIIColors.color_red, ASCIIColors.color_bright_green],
        )
        ASCIIColors.yellow(json.dumps(self.parameters, indent=4))
        self.fitted = True

    def fit(self, data: List[str]):
        """
        Ollama models do not require fitting as they are pre-trained models.
        This method is included to maintain consistency with the Vectorizer interface.

        Args:
            data (List[str]): The data to fit on (not used).
        """
        pass

    def vectorize(self, data: List[str]) -> List[np.ndarray]:
        """
        Vectorizes the input data using Ollama's embedding service.

        Args:
            data (List[str]): The data to vectorize.

        Returns:
            List[np.ndarray]: The list of Ollama embeddings for each input text.
        """
        if not self.fitted:
            ASCIIColors.error("OllamaVectorizer is not properly initialized.")
            return []

        embeddings = []
        try:
            for text in data:
                payload = {"model": self.model_name, "input": text}
                response = requests.post(self.base_url, json=payload)
                response.raise_for_status()
                embedding = np.array(response.json()["embeddings"][0])
                embeddings.append(embedding)
        except Exception as ex:
            trace_exception(ex)
            ASCIIColors.error("Failed to generate embeddings using Ollama API.")
        return embeddings

    def get_models(self):
        """
        Returns a list of available Ollama model names for embeddings.
        Note: This method should be implemented to fetch available models from the Ollama server.
        For now, it returns a static list as an example.
        """
        return ["bge-m3", "all-minilm", "nomic-embed-text"]

    def __str__(self):
        return f"Lollms Vector DB OllamaVectorizer. Using model {self.model_name} at {self.base_url}."

    def __repr__(self):
        return f"Lollms Vector DB OllamaVectorizer. Using model {self.model_name} at {self.base_url}."
