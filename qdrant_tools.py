"""
Qdrant Tools for MSP360 Documentation Search
Functions for searching and retrieving documentation from Qdrant with caching
"""

import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from embedding_service import EmbeddingService
from cache_service import get_search_cache

load_dotenv()

# Stopwords for keyword extraction in hybrid search
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'i', 'my', 'me', 'we', 'our', 'you',
    'your', 'he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their'
}


class QdrantTools:
    """Tools for interacting with Qdrant vector database with caching"""
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None,
        embedding_service: EmbeddingService = None,
        use_cache: bool = True
    ):
        """
        Initialize Qdrant tools.
        
        Args:
            url: Qdrant URL (defaults to env var)
            api_key: Qdrant API key (defaults to env var)
            collection_name: Collection name (defaults to msp360_docs_v2)
            embedding_service: Embedding service instance
            use_cache: Whether to use search result caching (default True)
        """
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "msp360_docs_v2")
        self.use_cache = use_cache
        self._cache = get_search_cache() if use_cache else None
        
        # Use prefer_grpc=False to avoid gRPC event loop issues
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key if self.api_key else None,
            prefer_grpc=False,
            timeout=60
        )
        
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize reranker for improved search relevance
        # Uses a lightweight cross-encoder model (~80MB, cached after first download)
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            max_length=512
        )
    
    def _format_result(self, result, score: float = None) -> Dict:
        """
        Format a single Qdrant result into a standard dictionary.
        
        Args:
            result: Qdrant search/scroll result object
            score: Optional score override (defaults to result.score or 1.0)
        
        Returns:
            Formatted dictionary with document metadata
        """
        url = result.payload.get("url", "")
        source = result.payload.get("source")
        if not source:
            source = "KB" if "kb.msp360.com" in url else "Help" if "help.msp360.com" in url else "Docs"
        
        return {
            "score": score if score is not None else getattr(result, 'score', 1.0),
            "url": url,
            "title": result.payload.get("title"),
            "text": result.payload.get("text"),
            "category": result.payload.get("category"),
            "error_code": result.payload.get("error_code"),
            "doc_type": result.payload.get("doc_type"),
            "source": source
        }
    
    def _boost_keyword_matches(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Boost results containing exact query terms (hybrid search).
        
        Args:
            results: List of formatted search results
            query: Original user query
        
        Returns:
            Results with updated hybrid_score based on keyword matches
        """
        # Tokenize query, filter stopwords and short terms
        query_terms = [
            t.lower() for t in query.split() 
            if t.lower() not in STOPWORDS and len(t) > 2
        ]
        
        for result in results:
            text = (result.get("text", "") + " " + result.get("title", "")).lower()
            
            # Count term matches
            matches = sum(1 for term in query_terms if term in text)
            
            # Boost: 0.03 per matching term (tunable)
            result["keyword_boost"] = matches * 0.03
            result["hybrid_score"] = result["score"] + result["keyword_boost"]
        
        return results
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank results using cross-encoder for better relevance.
        
        The cross-encoder evaluates query-document pairs together,
        providing more accurate relevance scores than embedding similarity.
        
        Args:
            query: User's search query
            results: List of formatted search results
            top_k: Number of top results to return
        
        Returns:
            Reranked and truncated list of results
        """
        if not results or len(results) <= 1:
            return results
        
        # Prepare query-document pairs (truncate text for efficiency)
        pairs = [(query, r.get("text", "")[:500]) for r in results]
        
        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for i, result in enumerate(results):
            result["rerank_score"] = float(rerank_scores[i])
        
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    
    async def search_docs(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
        use_reranking: bool = True,
        bypass_cache: bool = False
    ) -> List[Dict]:
        """
        Search documentation using semantic similarity with hybrid search, reranking, and caching.
        
        Uses a three-stage pipeline:
        1. Semantic search via embeddings (fetches 2x limit for reranking)
        2. Keyword boost (hybrid search) - boosts results containing exact query terms
        3. Cross-encoder reranking - reorders by query-document relevance
        
        Args:
            query: User's search query
            category: Filter by category (optional)
            limit: Number of results to return
            use_reranking: Whether to apply cross-encoder reranking (default True)
            bypass_cache: Whether to bypass the cache (default False)
        
        Returns:
            List of matching documents with metadata, sorted by relevance
        """
        # Check cache first (only if reranking enabled and cache not bypassed)
        if self._cache and use_reranking and not bypass_cache:
            cached = self._cache.get(query, category, limit)
            if cached is not None:
                return cached
        
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(
            query,
            task_type="retrieval_query"
        )
        
        # Build filter if category specified
        query_filter = None
        if category:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            )
        
        # Fetch more results for reranking (2x limit)
        fetch_limit = limit * 2 if use_reranking else limit
        
        # Search Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=fetch_limit,
            query_filter=query_filter
        )
        
        # Format results using helper method
        formatted_results = [self._format_result(r) for r in results]
        
        # Apply keyword boost (hybrid search)
        formatted_results = self._boost_keyword_matches(formatted_results, query)
        
        # Rerank if enabled and we have multiple results
        if use_reranking and len(formatted_results) > 1:
            formatted_results = self._rerank_results(query, formatted_results, top_k=limit)
        else:
            # Just sort by hybrid score and limit
            formatted_results = sorted(
                formatted_results,
                key=lambda x: x.get("hybrid_score", x["score"]),
                reverse=True
            )[:limit]
        
        # Cache the results
        if self._cache and use_reranking and not bypass_cache:
            self._cache.set(query, formatted_results, category, limit)
        
        return formatted_results
    
    async def get_page_by_url(self, url: str) -> Optional[Dict]:
        """
        Retrieve a specific page by URL.
        
        Args:
            url: The documentation page URL
        
        Returns:
            Document data or None if not found
        """
        # Scroll through all points with matching URL
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="url",
                        match=MatchValue(value=url)
                    )
                ]
            ),
            limit=1
        )[0]  # Get records from tuple
        
        if not results:
            return None
        
        # Return the first (should be only) result using helper
        return self._format_result(results[0])
    
    async def search_by_error_code(self, error_code: str, limit: int = 5) -> List[Dict]:
        """
        Search for documents by error code.
        
        Args:
            error_code: Error code to search for (e.g., "1531")
            limit: Number of results to return
        
        Returns:
            List of matching documents, with title matches boosted
        """
        # Clean error code (remove 'code' prefix if present)
        clean_code = error_code.replace("code", "").strip()
        
        # Search by error_code field
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="error_code",
                        match=MatchValue(value=clean_code)
                    )
                ]
            ),
            limit=limit
        )[0]
        
        # Format results using helper (score=1.0 for exact error code match)
        formatted_results = [self._format_result(r, score=1.0) for r in results]
        
        # Boost results with error code in title
        for result in formatted_results:
            if clean_code in (result.get("title") or ""):
                result["score"] = 1.1  # Boost exact title matches
        
        return sorted(formatted_results, key=lambda x: x["score"], reverse=True)[:limit]
    
    async def list_pages(self, category: Optional[str] = None) -> List[str]:
        """
        List all unique documentation page URLs.
        
        Args:
            category: Filter by category (optional)
        
        Returns:
            List of unique URLs
        """
        # Build filter
        query_filter = None
        if category:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            )
        
        # Scroll through all points (increased limit for larger collections)
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=100000,
            with_payload=["url", "title"]
        )[0]
        
        # Extract URLs with titles
        pages = []
        for r in results:
            url = r.payload.get("url")
            title = r.payload.get("title", "")
            if url:
                pages.append(f"{title} - {url}" if title else url)
        
        return sorted(list(set(pages)))
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.
        
        Returns:
            Collection metadata and statistics
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection.points_count,
                "vectors_count": collection.vectors_count,
                "status": collection.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_categories(self) -> List[str]:
        """
        Get list of all available categories.
        
        Returns:
            List of unique categories
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=100000,
            with_payload=["category"]
        )[0]
        
        categories = list(set([r.payload.get("category") for r in results if r.payload.get("category")]))
        return sorted(categories)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if self._cache:
            return self._cache.get_stats()
        return {"caching_disabled": True}


# Convenience functions
async def search_msp_docs(
    query: str,
    category: Optional[str] = None,
    limit: int = 5
) -> List[Dict]:
    """
    Quick function to search MSP360 documentation.
    
    Args:
        query: Search query
        category: Optional category filter
        limit: Number of results
    
    Returns:
        List of matching documents
    """
    tools = QdrantTools()
    return await tools.search_docs(query, category, limit)


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results for display to LLM.
    
    Args:
        results: List of search results
    
    Returns:
        Formatted string with results
    """
    if not results:
        return "No relevant documentation found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        error_code_str = ""
        if result.get("error_code"):
            error_code_str = f"\n**Error Code:** {result.get('error_code')}"
        
        source_str = f" [{result.get('source', 'Docs')}]" if result.get('source') else ""
        
        formatted.append(f"""
## Result {i} (Score: {result.get('score', 0):.4f}){source_str}
**Title:** {result.get('title', 'Unknown')}
**URL:** {result.get('url', 'Unknown')}
**Category:** {result.get('category', 'Unknown')}{error_code_str}

**Content:**
{result.get('text', 'No content available')[:2000]}{'...' if len(result.get('text', '')) > 2000 else ''}

---
""")
    
    return "\n".join(formatted)
