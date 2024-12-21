from abc import ABC, abstractmethod
import aiohttp
import asyncio
from typing import List, Union, AsyncGenerator
from pathlib import Path
from enum import Enum
from lollmsvectordb.remote_rag_database import SearchMode, LollmsRagDatabase


class LollmsLightRagConnector(LollmsRagDatabase):
    """LightRAG-specific implementation of the RAG database interface"""
    
    def __init__(self, base_url: str = "http://localhost:9621"):
        """Initialize the connector with the base URL of the LightRAG service"""
        self.base_url = base_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()

    async def ensure_session(self):
        """Ensure there's an active session"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def query(self, query_text: str, mode: SearchMode = SearchMode.hybrid, stream: bool = False) -> Union[str, AsyncGenerator]:
        await self.ensure_session()
        endpoint = f"{self.base_url}/query/stream" if stream else f"{self.base_url}/query"
        payload = {
            "query": query_text,
            "mode": mode,
            "stream": stream
        }
        async with self.session.post(endpoint, json=payload) as response:
            response.raise_for_status()
            if stream:
                async for chunk in response.content:
                    yield chunk.decode()
            else:
                result = await response.json()
                return result["response"]

    async def upload_file(self, file_path: Union[str, Path], description: str = None) -> dict:
        await self.ensure_session()
        file_path = Path(file_path)
        data = aiohttp.FormData()
        data.add_field('file', open(file_path, 'rb'), filename=file_path.name)
        if description:
            data.add_field('description', description)
        async with self.session.post(f"{self.base_url}/documents/file", data=data) as response:
            response.raise_for_status()
            return await response.json()

    async def upload_batch(self, file_paths: List[Union[str, Path]]) -> dict:
        await self.ensure_session()
        data = aiohttp.FormData()
        for file_path in file_paths:
            file_path = Path(file_path)
            data.add_field('files', open(file_path, 'rb'), filename=file_path.name)
        async with self.session.post(f"{self.base_url}/documents/batch", data=data) as response:
            response.raise_for_status()
            return await response.json()

    async def insert_text(self, text: str, description: str = None) -> dict:
        await self.ensure_session()
        payload = {
            "text": text,
            "description": description
        }
        async with self.session.post(f"{self.base_url}/documents/text", json=payload) as response:
            response.raise_for_status()
            return await response.json()

    async def scan_documents(self) -> dict:
        await self.ensure_session()
        async with self.session.post(f"{self.base_url}/documents/scan") as response:
            response.raise_for_status()
            return await response.json()

    async def clear_documents(self) -> dict:
        await self.ensure_session()
        async with self.session.delete(f"{self.base_url}/documents") as response:
            response.raise_for_status()
            return await response.json()

    async def health_check(self) -> dict:
        await self.ensure_session()
        async with self.session.get(f"{self.base_url}/health") as response:
            response.raise_for_status()
            return await response.json()