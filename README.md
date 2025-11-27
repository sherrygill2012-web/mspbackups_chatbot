# MSP360 Backup AI Documentation Chatbot

An intelligent chatbot that answers MSP360 Backup questions using RAG (Retrieval Augmented Generation) with documentation from help.msp360.com stored in Qdrant.

## ✨ Features

- **Semantic Search**: Uses Gemini embeddings to find relevant documentation
- **Error Code Search**: Quickly find solutions for specific error codes
- **Multiple Tools**: RAG search, error code lookup, page listing, content retrieval, category filtering
- **Smart Responses**: Cites sources and provides step-by-step solutions
- **Chat Interfaces**: CLI and Streamlit web UI with conversation history
- **Comprehensive Coverage**: Backup errors, troubleshooting, configuration, best practices

## 📋 Prerequisites

- Python 3.8+
- Qdrant running with `msp360_docs` collection
- OpenAI API key from OpenAI Platform

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and other settings
```

Required environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=msp360_docs
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/text-embedding-004
```

### 3. Run CLI Interface

```bash
python cli.py
```

### 4. Run Streamlit Web UI

```bash
streamlit run app.py
```

## 🏗️ Architecture

```
User Query
    ↓
MSP360 Expert Agent (Pydantic AI + OpenAI GPT-4o-mini)
    ↓
Agent Tools:
  1. retrieve_relevant_docs() - Semantic RAG search (Gemini embeddings)
  2. search_by_error_code() - Search by error code
  3. list_documentation_pages() - Browse available docs
  4. get_page_content() - Retrieve full page by URL
  5. get_available_categories() - List all categories
    ↓
Qdrant (msp360_docs collection with Gemini embeddings)
    ↓
OpenAI GPT-4o-mini LLM Response → Structured Answer with Citations
```

## 💬 Usage Examples

### CLI Interface

```bash
$ python cli.py

💾 MSP360 Backup Expert Assistant - CLI

💬 You: How to fix error code 1531?

🤖 Assistant: Error code 1531 occurs when synthetic full backup is not supported...
[Detailed explanation with solution and source URLs]

💬 You: What is Forever Forward Incremental backup?

🤖 Assistant: Forever Forward Incremental is a backup mechanism...
```

### Streamlit Web UI

1. Start the app: `streamlit run app.py`
2. Open your browser (usually http://localhost:8501)
3. Type your question in the chat input
4. Get instant answers with documentation sources

### Example Questions

- "How to fix error code 1531?"
- "What is synthetic full backup?"
- "How to configure Forever Forward Incremental?"
- "VSS error access denied"
- "How to restore a backup plan?"
- "What cloud storage supports synthetic backup?"
- "Explain backup retention policy"
- "Fix I/O error code 1076"

## 🛠️ Technical Details

### Agent Tools

1. **retrieve_relevant_docs(query, category, limit)**
   - Semantic search across all documentation
   - Optional category filtering
   - Returns top N relevant results

2. **search_by_error_code(error_code)**
   - Precise lookup by error code
   - Automatically cleans input (handles "code 1531", "1531", "error 1531")
   - Returns exact matches

3. **list_documentation_pages(category)**
   - Browse available documentation
   - Optional category filtering
   - Returns list of page URLs with titles

4. **get_page_content(url)**
   - Retrieve full page content
   - Returns complete documentation for specific URL

5. **get_available_categories()**
   - List all documentation categories
   - Helpful for exploration

### Data Model

The `msp360_docs` Qdrant collection uses the following metadata structure:

```python
{
    "url": str,          # Documentation URL
    "title": str,        # Page title
    "text": str,         # Full page content
    "category": str,     # Category (Backup, Restore, Errors, etc.)
    "error_code": str    # Error code if applicable (e.g., "1531")
}
```

### Configuration Options

Edit `.env` to customize:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `GEMINI_API_KEY`: Optional, for embeddings (uses OpenAI embeddings if not set)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `COLLECTION_NAME`: Collection name (default: msp360_docs)
- `LLM_PROVIDER`: LLM provider (default: openai; supports gemini/openai/anthropic/groq)
- `LLM_MODEL`: Model name (default: gpt-4o-mini)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)

## 📚 Documentation Categories

- **Backup**: Backup-related documentation and troubleshooting
- **Restore**: Restore procedures and issues
- **Errors**: Specific error codes and solutions
- **Warnings**: Warning messages and their meanings
- **Cloud Vendors**: Cloud storage provider configurations
- **Best Practices**: Backup strategy recommendations
- **Troubleshooting**: General troubleshooting guides

## 🔧 Development

### Project Structure

```
mspbackups_chatbot/
├── app.py                  # Streamlit web interface
├── cli.py                  # Command-line interface
├── msp_expert.py          # Pydantic AI agent with tools
├── qdrant_tools.py        # Qdrant search utilities
├── embedding_service.py   # Gemini embeddings wrapper
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

### Running Tests

```bash
# Test CLI
python cli.py

# Test Streamlit
streamlit run app.py

# Test individual components
python -c "from msp_expert import create_msp_expert; print('✓ Import successful')"
```

## 🐛 Troubleshooting

### Agent fails to initialize

- Check that `OPENAI_API_KEY` is set in `.env`
- Verify Qdrant is running: `curl http://localhost:6333`
- Confirm `msp360_docs` collection exists in Qdrant

### No results found

- Ensure the `msp360_docs` collection has data
- Check collection name matches in `.env`
- Verify embeddings were generated correctly during data ingestion

### Import errors

- Install all dependencies: `pip install -r requirements.txt`
- Use Python 3.8 or higher
- Check for conflicting package versions

## 📝 License

This project is created for MSP360 Backup documentation assistance.

## 🤝 Contributing

For questions or issues, please contact the development team.

## 📖 Resources

- [MSP360 Documentation](https://help.msp360.com)
- [MSP360 Knowledge Base](https://kb.msp360.com)
- [Pydantic AI Documentation](https://ai.pydantic.dev)
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [Gemini API](https://ai.google.dev)

