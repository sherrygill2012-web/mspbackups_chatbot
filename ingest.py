"""
Document Ingestion Pipeline for MSP360 RAG Chatbot
Handles scraping, processing, and indexing documentation into Qdrant
"""

import os
import re
import json
import hashlib
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from embedding_service import EmbeddingService

load_dotenv()


@dataclass
class Document:
    """Represents a documentation page"""
    url: str
    title: str
    text: str
    category: str
    error_code: Optional[str] = None
    doc_type: Optional[str] = None
    source: str = "Help"
    last_updated: Optional[str] = None
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload"""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "category": self.category,
            "error_code": self.error_code,
            "doc_type": self.doc_type,
            "source": self.source,
            "last_updated": self.last_updated or datetime.now().isoformat()
        }
    
    def generate_id(self) -> str:
        """Generate unique ID from URL"""
        return hashlib.md5(self.url.encode()).hexdigest()


class DocumentScraper:
    """Scrapes documentation from MSP360 help sites"""
    
    def __init__(
        self,
        base_urls: List[str] = None,
        max_concurrent: int = 5,
        delay_seconds: float = 0.5
    ):
        """
        Initialize scraper.
        
        Args:
            base_urls: Base URLs to scrape
            max_concurrent: Maximum concurrent requests
            delay_seconds: Delay between requests
        """
        self.base_urls = base_urls or [
            "https://help.msp360.com",
            "https://kb.msp360.com"
        ]
        self.max_concurrent = max_concurrent
        self.delay_seconds = delay_seconds
        self.visited_urls = set()
        self.documents: List[Document] = []
    
    def _extract_category(self, url: str, soup: BeautifulSoup) -> str:
        """Extract category from URL or page content"""
        path = urlparse(url).path.lower()
        
        if '/errors/' in path or '/error/' in path:
            return "Errors"
        elif '/warnings/' in path or '/warning/' in path:
            return "Warnings"
        elif '/restore/' in path:
            return "Restore"
        elif '/backup/' in path:
            return "Backup"
        elif '/cloud-vendors/' in path or '/storage/' in path:
            return "Cloud Vendors"
        elif '/best-practices/' in path:
            return "Best Practices"
        elif '/troubleshoot/' in path:
            return "Troubleshooting"
        elif '/managed-backup/' in path or '/mbs/' in path:
            return "Managed Backup Service"
        else:
            # Try to get from breadcrumb or navigation
            breadcrumb = soup.select_one('.breadcrumb, .nav-path, [class*="breadcrumb"]')
            if breadcrumb:
                text = breadcrumb.get_text().lower()
                if 'error' in text:
                    return "Errors"
                elif 'restore' in text:
                    return "Restore"
                elif 'backup' in text:
                    return "Backup"
            return "General"
    
    def _extract_error_code(self, url: str, title: str, text: str) -> Optional[str]:
        """Extract error code from page"""
        # Check URL
        url_match = re.search(r'/(\d{3,5})(?:[/-]|$)', url)
        if url_match:
            return url_match.group(1)
        
        # Check title
        title_match = re.search(r'\b(\d{3,5})\b', title)
        if title_match:
            return title_match.group(1)
        
        # Check first 200 chars of text
        text_match = re.search(r'(?:error\s*(?:code)?[:\s]*)?(\d{3,5})', text[:200], re.IGNORECASE)
        if text_match:
            return text_match.group(1)
        
        return None
    
    def _clean_text(self, soup: BeautifulSoup) -> str:
        """Extract and clean text from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get main content
        main_content = soup.select_one('main, article, .content, .article-content, #content')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[Document]:
        """Fetch and parse a single page"""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        
        try:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Get title
                title_elem = soup.select_one('h1, .article-title, title')
                title = title_elem.get_text(strip=True) if title_elem else urlparse(url).path
                
                # Clean title
                title = title.replace(' | MSP360', '').replace(' - MSP360', '').strip()
                
                # Get text content
                text = self._clean_text(soup)
                
                if len(text) < 100:  # Skip very short pages
                    return None
                
                # Determine category
                category = self._extract_category(url, soup)
                
                # Extract error code
                error_code = self._extract_error_code(url, title, text)
                
                # Determine source
                source = "KB" if "kb.msp360" in url else "Help"
                
                return Document(
                    url=url,
                    title=title,
                    text=text,
                    category=category,
                    error_code=error_code,
                    source=source
                )
                
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    async def _discover_links(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> List[str]:
        """Discover links from a page"""
        try:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    
                    # Make absolute URL
                    if not href.startswith('http'):
                        href = urljoin(url, href)
                    
                    # Filter to same domain
                    parsed = urlparse(href)
                    if any(base in href for base in self.base_urls):
                        # Skip non-doc pages
                        if not any(x in href.lower() for x in [
                            '/login', '/signup', '/cart', '/download',
                            '.pdf', '.zip', '.exe', '#', 'javascript:'
                        ]):
                            links.append(href.split('#')[0])  # Remove anchor
                
                return list(set(links))
                
        except Exception as e:
            print(f"Error discovering links from {url}: {e}")
            return []
    
    async def scrape(
        self,
        start_urls: List[str] = None,
        max_pages: int = 1000
    ) -> List[Document]:
        """
        Scrape documentation.
        
        Args:
            start_urls: URLs to start from (uses base_urls if not provided)
            max_pages: Maximum pages to scrape
        
        Returns:
            List of scraped documents
        """
        start_urls = start_urls or self.base_urls
        to_visit = list(start_urls)
        
        async with aiohttp.ClientSession() as session:
            while to_visit and len(self.documents) < max_pages:
                # Process batch
                batch = to_visit[:self.max_concurrent]
                to_visit = to_visit[self.max_concurrent:]
                
                # Fetch pages
                tasks = [self._fetch_page(session, url) for url in batch]
                results = await asyncio.gather(*tasks)
                
                for doc in results:
                    if doc:
                        self.documents.append(doc)
                        print(f"Scraped: {doc.title[:50]}... ({len(self.documents)} total)")
                
                # Discover more links
                for url in batch:
                    if url not in self.visited_urls:
                        new_links = await self._discover_links(session, url)
                        to_visit.extend([l for l in new_links if l not in self.visited_urls])
                
                # Delay
                await asyncio.sleep(self.delay_seconds)
        
        return self.documents


class DocumentIndexer:
    """Indexes documents into Qdrant"""
    
    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = None,
        embedding_service: EmbeddingService = None
    ):
        """
        Initialize indexer.
        
        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name
            embedding_service: Embedding service instance
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "msp360_docs")
        self.embedding_service = embedding_service or EmbeddingService()
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False
        )
    
    def ensure_collection(self, recreate: bool = False):
        """Ensure collection exists with correct schema"""
        exists = self.client.collection_exists(self.collection_name)
        
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.get_embedding_dimension(),
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection exists: {self.collection_name}")
    
    async def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 10
    ) -> int:
        """
        Index documents into Qdrant.
        
        Args:
            documents: Documents to index
            batch_size: Batch size for indexing
        
        Returns:
            Number of indexed documents
        """
        indexed = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            points = []
            for doc in batch:
                # Generate embedding
                embedding = await self.embedding_service.generate_embedding(
                    doc.text[:8000],  # Truncate for embedding
                    task_type="retrieval_document"
                )
                
                point = PointStruct(
                    id=doc.generate_id(),
                    vector=embedding,
                    payload=doc.to_payload()
                )
                points.append(point)
            
            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            indexed += len(batch)
            print(f"Indexed {indexed}/{len(documents)} documents")
        
        return indexed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status
        }


async def run_ingestion(
    start_urls: List[str] = None,
    max_pages: int = 500,
    recreate_collection: bool = False
):
    """
    Run the full ingestion pipeline.
    
    Args:
        start_urls: URLs to start scraping from
        max_pages: Maximum pages to scrape
        recreate_collection: Whether to recreate the collection
    """
    print("=" * 60)
    print("MSP360 Documentation Ingestion Pipeline")
    print("=" * 60)
    
    # Scrape documents
    print("\n[1/3] Scraping documentation...")
    scraper = DocumentScraper()
    documents = await scraper.scrape(start_urls=start_urls, max_pages=max_pages)
    print(f"Scraped {len(documents)} documents")
    
    # Save backup
    backup_path = f"scraped_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, 'w') as f:
        json.dump([asdict(d) for d in documents], f, indent=2)
    print(f"Saved backup to {backup_path}")
    
    # Index documents
    print("\n[2/3] Indexing documents...")
    indexer = DocumentIndexer()
    indexer.ensure_collection(recreate=recreate_collection)
    indexed = await indexer.index_documents(documents)
    print(f"Indexed {indexed} documents")
    
    # Print stats
    print("\n[3/3] Collection statistics:")
    stats = indexer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MSP360 Documentation Ingestion Pipeline")
    parser.add_argument("--urls", nargs="+", help="Start URLs for scraping")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages to scrape")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    
    args = parser.parse_args()
    
    asyncio.run(run_ingestion(
        start_urls=args.urls,
        max_pages=args.max_pages,
        recreate_collection=args.recreate
    ))

