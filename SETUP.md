# MSP360 Backup Chatbot - Setup Guide

## Quick Start

### 1. Create Virtual Environment

```bash
cd mspbackups_chatbot
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # On Windows PowerShell
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Create .env file from example
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

Required configuration:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=msp_docs_v2
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
```

### 4. Verify Qdrant Collection

Make sure your `msp_docs_v2` collection is populated with data from both:
- kb.msp360.com
- help.msp360.com

### 5. Run the Chatbot

**CLI Mode:**
```bash
python cli.py
```

**Web UI Mode:**
```bash
streamlit run app.py
```

## Features

✅ **Handles Multiple Data Sources**
- Automatically detects and labels sources (KB vs Help)
- Results show [KB] or [Help] tags for easy identification

✅ **Optimized for Large Collections**
- Supports up to 100,000 documents
- Efficient semantic search
- Smart result ranking

✅ **Error Code Search**
- Quick lookup by error code (e.g., "1531")
- Automatic cleaning of input formats

✅ **Category Filtering**
- Filter by documentation category
- Browse by topic area

## Code Enhancements for Multiple Sources

The chatbot automatically:

1. **Detects source from URL**
   - `kb.msp360.com` → labeled as [KB]
   - `help.msp360.com` → labeled as [Help]

2. **Displays source in results**
   - Each search result shows which site it came from
   - Example: "Result 1 (Score: 0.9234) [KB]"

3. **Handles larger collections**
   - Scroll limit increased to 100,000 documents
   - Efficient pagination and filtering

## Example Usage

### Error Code Search
```
You: How to fix error 1531?
Assistant: [Searches and returns KB article with solution]
```

### General Question
```
You: What is synthetic full backup?
Assistant: [Searches both KB and Help docs, returns best matches]
```

### Browse Documentation
```
You: List all backup-related pages
Assistant: [Returns pages from both sources]
```

## Troubleshooting

### "Agent failed to initialize"
- Check OPENAI_API_KEY is set correctly
- Verify Qdrant is running: `curl http://localhost:6333`
- Confirm msp_docs_v2 collection exists

### "No results found"
- Verify collection has data: Check Qdrant dashboard
- Try broader search terms
- Check collection name in .env matches actual collection

### Import errors
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check Python version: `python --version` (need 3.8+)

## Next Steps

Once running locally, you can:
- Test with various queries
- Adjust search parameters in the code
- Add more documentation sources
- Deploy to production environment

