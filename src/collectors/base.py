"""Base collector with HTTP client and retry logic."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup

from src.config import MAX_RETRIES, RATE_LIMIT_DELAY, REQUEST_TIMEOUT, USER_AGENT


class BaseCollector(ABC):
    """Base class for data collectors with HTTP retry logic."""

    def __init__(self):
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        self.last_request_time = 0.0

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    async def fetch_json(
        self, url: str, retries: int = MAX_RETRIES
    ) -> Optional[dict]:
        """Fetch JSON from URL with retry logic."""
        await self._rate_limit()

        for attempt in range(retries):
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        return None

    async def fetch_html(self, url: str, retries: int = MAX_RETRIES) -> Optional[BeautifulSoup]:
        """Fetch HTML from URL and parse with BeautifulSoup."""
        await self._rate_limit()

        for attempt in range(retries):
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                return BeautifulSoup(response.text, "lxml")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        return None

    async def fetch_csv(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """Fetch CSV content from URL."""
        await self._rate_limit()

        for attempt in range(retries):
            try:
                response = await self.client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return None
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise

        return None

    @abstractmethod
    async def collect(self) -> Any:
        """Collect data. Must be implemented by subclasses."""
        pass
