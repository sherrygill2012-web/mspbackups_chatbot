# MSP360 Backup AI Documentation Chatbot

An intelligent chatbot that answers MSP360 Backup questions using RAG (Retrieval Augmented Generation) with documentation from help.msp360.com stored in Qdrant.

## Features

### Core Capabilities
- **Semantic Search**: Uses Gemini/OpenAI embeddings to find relevant documentation
- **Error Code Search**: Quickly find solutions for specific error codes
- **Multiple Tools**: RAG search, error code lookup, page listing, content retrieval, category filtering
- **Smart Responses**: Cites sources and provides step-by-step solutions
- **Multi-Provider LLM**: Supports OpenAI, Gemini, Anthropic, and Groq

### Advanced Features
- **Query Expansion**: Multi-query retrieval for better coverage
- **Hybrid Search**: Combines semantic search with keyword boosting
- **Cross-Encoder Reranking**: Improved result relevance
- **Response Caching**: Reduced latency for repeated queries
- **Streaming Responses**: Real-time response generation

### Interfaces
- **Streamlit Web UI**: Beautiful chat interface with dark mode
- **CLI**: Command-line interface for testing
- **REST API**: FastAPI-based API for programmatic access
- **Slack Bot**: Answer questions directly in Slack

### Analytics & Monitoring
- **Analytics Dashboard**: Track queries, response times, and feedback
- **Feedback System**: Thumbs up/down for response quality tracking
- **Cache Statistics**: Monitor cache hit rates
- **Knowledge Gap Detection**: Identify unanswered queries

## Prerequisites

- Python 3.8+
- Qdrant running with `msp360_docs_v2` collection
- API keys for your chosen providers (OpenAI, Gemini, etc.)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env_example.txt .env
# Edit .env with your API keys and settings
```

Required environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=msp360_docs_v2
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
```

### 3. Run Streamlit Web UI

```bash
streamlit run app.py
```

### 4. Run CLI Interface

```bash
python cli.py
```

### 5. Run REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6. Run Slack Bot

```bash
python slack_bot.py
```

## Architecture

```
User Query
    ↓
MSP360 Expert Agent (Pydantic AI)
    ↓
Query Enhancement (Multi-Query Expansion)
    ↓
Agent Tools:
  1. retrieve_relevant_docs() - Semantic RAG search
  2. enhanced_multi_query_search() - Multi-query retrieval
  3. search_by_error_code() - Error code lookup
  4. list_documentation_pages() - Browse available docs
  5. get_page_content() - Retrieve full page by URL
  6. get_available_categories() - List all categories
  7. search_related_topics() - Find related documentation
    ↓
Qdrant (with hybrid search + reranking)
    ↓
Caching Layer (embeddings + search results)
    ↓
LLM Response → Structured Answer with Citations
```

## API Reference

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Send chat message |
| `/chat/stream` | POST | Stream chat response |
| `/search` | POST | Search documentation |
| `/error-code` | POST | Look up error code |
| `/categories` | GET | List categories |
| `/feedback` | POST | Submit feedback |
| `/cache/stats` | GET | Cache statistics |
| `/cache/clear` | POST | Clear caches |

### Example API Usage

```python
import requests

# Chat endpoint
response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "How to fix error code 1531?"}
)
print(response.json()["response"])

# Search endpoint
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "VSS backup", "limit": 5}
)
for result in response.json()["results"]:
    print(f"- {result['title']}: {result['url']}")
```

## Project Structure

```
mspbackups_chatbot/
├── app.py                  # Streamlit web interface
├── cli.py                  # Command-line interface
├── api.py                  # FastAPI REST API
├── msp_expert.py          # Pydantic AI agent with tools
├── qdrant_tools.py        # Qdrant search utilities
├── embedding_service.py   # Embeddings wrapper with caching
├── cache_service.py       # Caching layer (embeddings + search)
├── analytics.py           # Analytics tracking service
├── slack_bot.py           # Slack bot integration
├── image_processor.py     # Multi-modal image processing
├── ingest.py              # Document ingestion pipeline
├── pages/
│   └── 1_Analytics_Dashboard.py  # Streamlit analytics page
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker container
├── docker-entrypoint.sh  # Docker entrypoint
├── env_example.txt       # Environment template
└── README.md             # This file
```

## Docker Deployment

### Build Image

```bash
docker build -t msp360-chatbot .
```

### Run Streamlit (default)

```bash
docker run -p 8501:8501 --env-file .env msp360-chatbot
```

### Run API Server

```bash
docker run -p 8000:8000 --env-file .env -e RUN_MODE=api msp360-chatbot
```

### Run Slack Bot

```bash
docker run --env-file .env -e RUN_MODE=slack msp360-chatbot
```

## Kubernetes Deployment

See the `k8s/` directory for Kubernetes manifests:
- `deployment.yaml` - Main deployment
- `service.yaml` - Service configuration
- `configmap.yaml` - Configuration
- `secret.yaml` - Secrets (API keys)
- `argocd-application.yaml` - ArgoCD application

## Document Ingestion

To update the documentation index:

```bash
# Scrape and index documentation
python ingest.py --max-pages 500

# Recreate collection from scratch
python ingest.py --recreate --max-pages 500

# Start from specific URLs
python ingest.py --urls https://help.msp360.com/backup/errors
```

## Configuration Options

### Caching

```bash
EMBEDDING_CACHE_SIZE=5000      # Max cached embeddings
EMBEDDING_CACHE_TTL=86400      # Cache TTL (24 hours)
SEARCH_CACHE_SIZE=1000         # Max cached searches
SEARCH_CACHE_TTL=1800          # Cache TTL (30 minutes)
```

### Rate Limiting (API)

```bash
RATE_LIMIT_REQUESTS=60         # Requests per window
RATE_LIMIT_WINDOW=60           # Window in seconds
```

### Analytics

```bash
ANALYTICS_PERSIST_PATH=analytics_data.json
```

## Example Questions

- "How to fix error code 1531?"
- "What is synthetic full backup?"
- "How to configure Forever Forward Incremental?"
- "VSS error access denied"
- "How to restore a backup plan?"
- "What cloud storage supports synthetic backup?"
- "Explain backup retention policy"
- "Fix I/O error code 1076"

## Troubleshooting

### Agent fails to initialize

- Check that required API keys are set in `.env`
- Verify Qdrant is running: `curl http://localhost:6333`
- Confirm `msp360_docs_v2` collection exists in Qdrant

### No results found

- Ensure the collection has data
- Check collection name matches in `.env`
- Verify embeddings model matches the indexed documents

### Slow responses

- Enable caching (enabled by default)
- Check Qdrant performance
- Consider reducing `limit` parameter

### Import errors

- Install all dependencies: `pip install -r requirements.txt`
- Use Python 3.8 or higher

## License

This project is created for MSP360 Backup documentation assistance.

## Contributing

For questions or issues, please contact the development team.

## Resources

- [MSP360 Documentation](https://help.msp360.com)
- [MSP360 Knowledge Base](https://kb.msp360.com)
- [Pydantic AI Documentation](https://ai.pydantic.dev)
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
