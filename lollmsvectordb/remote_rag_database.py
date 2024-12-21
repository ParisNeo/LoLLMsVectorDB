from abc import ABC, abstractmethod
import aiohttp
import asyncio
from typing import List, Union, AsyncGenerator
from pathlib import Path
from enum import Enum

class SearchMode(str, Enum):
    naive = "naive"
    local = "local" 
    global_ = "global"
    hybrid = "hybrid"

class LollmsRagDatabase(ABC):
    """Abstract base interface for RAG database operations"""
    
    @abstractmethod
    async def __aenter__(self):
        """Context manager entry"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass

    @abstractmethod
    async def query(self, query_text: str, mode: SearchMode = SearchMode.hybrid, stream: bool = False) -> Union[str, AsyncGenerator]:
        """Query the RAG system"""
        pass

    @abstractmethod
    async def upload_file(self, file_path: Union[str, Path], description: str = None) -> dict:
        """Upload a single file to the RAG system"""
        pass

    @abstractmethod
    async def upload_batch(self, file_paths: List[Union[str, Path]]) -> dict:
        """Upload multiple files in batch"""
        pass

    @abstractmethod
    async def insert_text(self, text: str, description: str = None) -> dict:
        """Insert text directly into the RAG system"""
        pass

    @abstractmethod
    async def scan_documents(self) -> dict:
        """Trigger a scan for new documents"""
        pass

    @abstractmethod
    async def clear_documents(self) -> dict:
        """Clear all documents from the RAG system"""
        pass

    @abstractmethod
    async def health_check(self) -> dict:
        """Check the health status of the RAG system"""
        pass
