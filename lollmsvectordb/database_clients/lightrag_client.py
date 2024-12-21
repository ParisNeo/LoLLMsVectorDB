from typing import List, Union
from pathlib import Path
from lollmsvectordb.remote_rag_database import SearchMode, LollmsRagDatabase
import requests
class LollmsLightRagConnector(LollmsRagDatabase):
    """LightRAG-specific implementation of the RAG database interface"""
    
    def __init__(self, base_url: str = "http://localhost:9621"):
        """Initialize the connector with the base URL of the LightRAG service"""
        self.base_url = base_url.rstrip('/')

    def query(self, query_text: str, mode: SearchMode = SearchMode.hybrid, stream: bool = False) -> Union[str, bytes]:
        endpoint = f"{self.base_url}/query/stream" if stream else f"{self.base_url}/query"
        payload = {
            "query": query_text,
            "mode": mode,
            "stream": stream
        }
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        if stream:
            return response.content  # Return raw bytes for streaming
        else:
            return response.json()["response"]

    def upload_file(self, file_path: Union[str, Path], description: str = None) -> dict:
        file_path = Path(file_path)
        data = {'file': open(file_path, 'rb')}
        if description:
            data['description'] = description
        response = requests.post(f"{self.base_url}/documents/file", files=data)
        response.raise_for_status()
        return response.json()

    def upload_batch(self, file_paths: List[Union[str, Path]]) -> dict:
        files = [('files', open(Path(file_path), 'rb')) for file_path in file_paths]
        response = requests.post(f"{self.base_url}/documents/batch", files=files)
        response.raise_for_status()
        return response.json()

    def insert_text(self, text: str, description: str = None) -> dict:
        payload = {
            "text": text,
            "description": description
        }
        response = requests.post(f"{self.base_url}/documents/text", json=payload)
        response.raise_for_status()
        return response.json()

    def scan_documents(self) -> dict:
        response = requests.post(f"{self.base_url}/documents/scan")
        response.raise_for_status()
        return response.json()

    def clear_documents(self) -> dict:
        response = requests.delete(f"{self.base_url}/documents")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()