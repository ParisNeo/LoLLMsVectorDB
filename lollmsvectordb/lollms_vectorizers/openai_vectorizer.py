"""
LoLLMsVectorDB

File: openai_vectorizer.py
Author: ParisNeo
Description: Contains the OpenAIVectorizer class for vectorizing text data using OpenAI's embedding service.

This file is part of the LoLLMsVectorDB project, a modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.
"""
from ascii_colors import ASCIIColors, trace_exception
import openai
import numpy as np
from lollmsvectordb.vectorizer import Vectorizer
from typing import List
import os
import json

class OpenAIVectorizer(Vectorizer):
    def __init__(self, model_name: str = 'text-embedding-ada-002', api_key: str|None = None):
        """
        Initializes the OpenAIVectorizer with a specified OpenAI model.

        Args:
            api_key (str): The API key for accessing OpenAI services. If not provided, it will use the environment variable 'OPENAI_API_KEY'.
            model_name (str): The name of the OpenAI model to use for embeddings.
        """
        super().__init__("OpenAIVectorizer")
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

        if not self.api_key:
            ASCIIColors.error("OpenAI API key is missing. Please provide it either in the constructor or set it as an environment variable 'OPENAI_API_KEY'.")
            self.fitted = False
        else:
            self.parameters = {
                "api_key": self.api_key,
                "model_name": self.model_name
            }
            ASCIIColors.multicolor(["LollmsVectorDB>", f"Using OpenAI model {model_name} for embeddings."], [ASCIIColors.color_red, ASCIIColors.color_cyan], end="", flush=True)
            ASCIIColors.success("OK")
            ASCIIColors.multicolor(["LollmsVectorDB>", f" Parameters:"], [ASCIIColors.color_red, ASCIIColors.color_bright_green])
            ASCIIColors.yellow(json.dumps(self.parameters, indent=4))
            self.fitted = True

    def fit(self, data: List[str]):
        """
        OpenAI models do not require fitting as they are pre-trained models.
        This method is included to maintain consistency with the Vectorizer interface.

        Args:
            data (List[str]): The data to fit on (not used).
        """
        pass

    def vectorize(self, data: List[str]) -> List[np.ndarray]:
        """
        Vectorizes the input data using OpenAI's embedding service.

        Args:
            data (List[str]): The data to vectorize.

        Returns:
            List[np.ndarray]: The list of OpenAI embeddings for each input text.
        """
        if not self.fitted:
            ASCIIColors.error("OpenAIVectorizer is not properly initialized. Please check your API key.")
            return []

        embeddings = []
        try:
            for text in data:
                response = openai.Embedding.create(
                    input=text,
                    model=self.model_name,
                    api_key=self.api_key
                )
                embedding = np.array(response['data'][0]['embedding'])
                embeddings.append(embedding)
        except Exception as ex:
            trace_exception(ex)
            ASCIIColors.error("Failed to generate embeddings using OpenAI API.")
        return embeddings

    def get_models(self):
        """
        Returns a list of available OpenAI model names for embeddings.
        """
        return [
            "text-embedding-ada-002",
            "text-embedding-babbage-001",
            "text-embedding-curie-001",
            "text-embedding-davinci-001"
        ]

    def __str__(self):
        model_name = self.parameters['model_name']
        return f'Lollms Vector DB OpenAIVectorizer. Using model {model_name}.'

    def __repr__(self):
        model_name = self.parameters['model_name']
        return f'Lollms Vector DB OpenAIVectorizer. Using model {model_name}.'
