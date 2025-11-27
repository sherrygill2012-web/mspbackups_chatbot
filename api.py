"""
REST API for MSP360 Backup Expert
FastAPI-based API for programmatic access to the chatbot
"""

import os
import time
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from msp_expert import create_msp_expert, MSPDeps
from cache_service import get_all_cache_stats, clear_all_caches

load_dotenv()

# Rate limiting storage (in-memory, use Redis in production)
rate_limit_store: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# API Key validation (optional)
API_KEY = os.getenv("API_KEY", None)

# Global agent instance
agent = None
deps = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global agent, deps
    # Startup: Initialize agent
    print("Initializing MSP360 Expert Agent...")
    agent, deps = create_msp_expert()
    print("Agent ready!")
    yield
    # Shutdown: Cleanup
    print("Shutting down...")


app = FastAPI(
    title="MSP360 Backup Expert API",
    description="REST API for the MSP360 Backup documentation chatbot with RAG",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    category: Optional[str] = Field(None, description="Optional category filter")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Agent's response")
    query: str = Field(..., description="Original query")
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: str = Field(..., description="ISO timestamp")


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    category: Optional[str] = Field(None, description="Optional category filter")
    limit: int = Field(5, ge=1, le=20, description="Number of results")


class SearchResult(BaseModel):
    """Individual search result"""
    title: str
    url: str
    category: Optional[str]
    error_code: Optional[str]
    score: float
    text_snippet: str


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult]
    query: str
    total_results: int
    response_time: float


class ErrorCodeRequest(BaseModel):
    """Error code lookup request"""
    error_code: str = Field(..., description="Error code to look up")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    agent_ready: bool
    timestamp: str


class FeedbackRequest(BaseModel):
    """Feedback submission request"""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Agent's response")
    rating: str = Field(..., pattern="^(positive|negative)$", description="Rating: positive or negative")
    comment: Optional[str] = Field(None, max_length=500, description="Optional feedback comment")


class CacheStatsResponse(BaseModel):
    """Cache statistics response"""
    embedding_cache: Dict[str, Any]
    search_cache: Dict[str, Any]


# Rate limiting dependency
async def check_rate_limit(request: Request):
    """Check and enforce rate limiting"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    if client_ip in rate_limit_store:
        rate_limit_store[client_ip] = [
            t for t in rate_limit_store[client_ip]
            if current_time - t < RATE_LIMIT_WINDOW
        ]
    else:
        rate_limit_store[client_ip] = []
    
    # Check limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
        )
    
    # Record this request
    rate_limit_store[client_ip].append(current_time)


# API Key validation dependency
async def validate_api_key(x_api_key: Optional[str] = Header(None)):
    """Validate API key if configured"""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        agent_ready=agent is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: None = Depends(validate_api_key)
):
    """
    Send a chat message and get a response.
    
    The agent will search the MSP360 documentation and provide
    a helpful response with source citations.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time = time.time()
    
    try:
        result = await agent.run(request.query, deps=deps)
        response_time = time.time() - start_time
        
        return ChatResponse(
            response=result.data,
            query=request.query,
            response_time=round(response_time, 3),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: None = Depends(validate_api_key)
):
    """
    Send a chat message and stream the response.
    
    Returns a Server-Sent Events stream with the response chunks.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def generate():
        try:
            async with agent.run_stream(request.query, deps=deps) as result:
                async for text in result.stream_text(delta=True):
                    yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    request: SearchRequest,
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: None = Depends(validate_api_key)
):
    """
    Search the documentation directly without LLM processing.
    
    Returns raw search results with relevance scores.
    """
    if deps is None:
        raise HTTPException(status_code=503, detail="Dependencies not initialized")
    
    start_time = time.time()
    
    try:
        results = await deps.qdrant_tools.search_docs(
            query=request.query,
            category=request.category,
            limit=request.limit
        )
        
        search_results = [
            SearchResult(
                title=r.get("title", "Unknown"),
                url=r.get("url", ""),
                category=r.get("category"),
                error_code=r.get("error_code"),
                score=round(r.get("score", 0), 4),
                text_snippet=r.get("text", "")[:300] + "..." if len(r.get("text", "")) > 300 else r.get("text", "")
            )
            for r in results
        ]
        
        return SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            response_time=round(time.time() - start_time, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/error-code", tags=["Search"])
async def lookup_error_code(
    request: ErrorCodeRequest,
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: None = Depends(validate_api_key)
):
    """
    Look up documentation for a specific error code.
    """
    if deps is None:
        raise HTTPException(status_code=503, detail="Dependencies not initialized")
    
    try:
        results = await deps.qdrant_tools.search_by_error_code(request.error_code)
        
        return {
            "error_code": request.error_code,
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "category": r.get("category"),
                    "text": r.get("text", "")[:500]
                }
                for r in results
            ],
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", tags=["Search"])
async def get_categories(
    _api_key: None = Depends(validate_api_key)
):
    """
    Get list of available documentation categories.
    """
    if deps is None:
        raise HTTPException(status_code=503, detail="Dependencies not initialized")
    
    try:
        categories = await deps.qdrant_tools.get_categories()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection-info", tags=["System"])
async def get_collection_info(
    _api_key: None = Depends(validate_api_key)
):
    """
    Get information about the Qdrant collection.
    """
    if deps is None:
        raise HTTPException(status_code=503, detail="Dependencies not initialized")
    
    return deps.qdrant_tools.get_collection_info()


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    _api_key: None = Depends(validate_api_key)
):
    """
    Submit feedback on a response.
    
    In production, this would store feedback in a database for analysis.
    """
    # In production, store this in a database
    feedback_id = hashlib.md5(
        f"{request.query}{request.response}{time.time()}".encode()
    ).hexdigest()[:12]
    
    return {
        "status": "received",
        "feedback_id": feedback_id,
        "rating": request.rating,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["System"])
async def get_cache_stats(
    _api_key: None = Depends(validate_api_key)
):
    """
    Get cache statistics for embeddings and search results.
    """
    return get_all_cache_stats()


@app.post("/cache/clear", tags=["System"])
async def clear_cache(
    _api_key: None = Depends(validate_api_key)
):
    """
    Clear all caches (embeddings and search results).
    """
    clear_all_caches()
    return {"status": "cleared", "timestamp": datetime.now().isoformat()}


# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true"
    )

