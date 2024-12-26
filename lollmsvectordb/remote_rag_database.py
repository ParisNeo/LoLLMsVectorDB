import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, List, Union

import aiohttp


class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"


class LollmsRagDatabase(ABC):
    """Abstract base interface for RAG database operations"""

    @abstractmethod
    def query(
        self,
        query_text: str,
        mode: SearchMode = SearchMode.hybrid,
        stream: bool = False,
    ) -> Union[str, bytes]:
        """Query the RAG system"""
        pass

    @abstractmethod
    def upload_file(self, file_path: Union[str, Path], description: str = None) -> dict:
        """Upload a single file to the RAG system"""
        pass

    @abstractmethod
    def upload_batch(self, file_paths: List[Union[str, Path]]) -> dict:
        """Upload multiple files in batch"""
        pass

    @abstractmethod
    def insert_text(self, text: str, description: str = None) -> dict:
        """Insert text directly into the RAG system"""
        pass

    @abstractmethod
    def scan_documents(self) -> dict:
        """Trigger a scan for new documents"""
        pass

    @abstractmethod
    def clear_documents(self) -> dict:
        """Clear all documents from the RAG system"""
        pass

    @abstractmethod
    def health_check(self) -> dict:
        """Check the health status of the RAG system"""
        pass
