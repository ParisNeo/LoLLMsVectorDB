from pathlib import Path
from typing import List, Union

import requests

from lollmsvectordb.remote_rag_database import LollmsRagDatabase, SearchMode


class LollmsLightRagConnector(LollmsRagDatabase):
    """LightRAG-specific implementation of the RAG database interface"""

    def __init__(self, base_url: str = "http://localhost:9621", api_key: str = None):
        """Initialize the connector with the base URL of the LightRAG service and optional API key"""
        self.base_url = base_url.rstrip("/")
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def query(
        self,
        query_text: str,
        mode: SearchMode = SearchMode.hybrid,
        stream: bool = False,
        only_need_context: bool = True,
    ) -> Union[str, bytes]:
        endpoint = (
            f"{self.base_url}/query/stream" if stream else f"{self.base_url}/query"
        )
        payload = {
            "query": query_text,
            "mode": mode,
            "stream": stream,
            "only_need_context": only_need_context,
        }
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        if stream:
            return response.content
        else:
            return response.json()["response"]

    def upload_file(self, file_path: Union[str, Path], description: str = None) -> dict:
        file_path = Path(file_path)
        data = {"file": open(file_path, "rb")}
        if description:
            data["description"] = description
        response = requests.post(f"{self.base_url}/documents/file", files=data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def upload_batch(self, file_paths: List[Union[str, Path]]) -> dict:
        files = [("files", open(Path(file_path), "rb")) for file_path in file_paths]
        response = requests.post(f"{self.base_url}/documents/batch", files=files, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def insert_text(self, text: str, description: str = None) -> dict:
        payload = {"text": text, "description": description}
        response = requests.post(f"{self.base_url}/documents/text", json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def scan_documents(self) -> dict:
        response = requests.post(f"{self.base_url}/documents/scan", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def clear_documents(self) -> dict:
        response = requests.delete(f"{self.base_url}/documents", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict:
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()
