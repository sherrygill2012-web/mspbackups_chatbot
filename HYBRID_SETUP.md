# Hybrid Configuration: OpenAI LLM + Gemini Embeddings

Your MSP360 Backup Chatbot uses a **hybrid approach**:
- 🤖 **OpenAI GPT-4o-mini** for intelligent chat responses
- 🔍 **Gemini embeddings** for semantic search (matches your existing Qdrant collection)

## Why This Configuration?

Your Qdrant collection (`msp_docs_v2`) was created with **Gemini embeddings (768 dimensions)**. To use this existing collection, we must continue using Gemini for embeddings. However, we can still benefit from OpenAI's superior language model for generating responses!

## Setup Instructions

### 1. Get Both API Keys

#### OpenAI API Key (for LLM):
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-proj-` or `sk-`)

#### Gemini API Key (for embeddings):
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key

### 2. Create `.env` File

Create `mspbackups_chatbot/.env`:

```bash
# OpenAI for LLM (chat responses)
OPENAI_API_KEY=sk-proj-your_openai_key_here

# Gemini for embeddings (search)
GEMINI_API_KEY=your_gemini_key_here

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=msp_docs_v2

# LLM Configuration - use OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Embedding Configuration - use Gemini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
```

### 3. Install Dependencies

```bash
cd mspbackups_chatbot
source venv/bin/activate
pip install -r requirements.txt
```

This installs both `openai` and `google-generativeai` packages.

### 4. Run the Chatbot

```bash
# CLI mode
python cli.py

# Streamlit web UI
streamlit run app.py
```

## How It Works

### Query Flow:

1. **User asks a question** → "How to fix error 1531?"

2. **Gemini generates embedding** for the query (768 dimensions)
   - Uses `gemini-embedding-001` model
   - Matches your Qdrant collection format

3. **Qdrant searches** for similar documents
   - Semantic search using Gemini embeddings
   - Returns top 5 relevant chunks

4. **OpenAI GPT-4o-mini generates answer**
   - Receives documentation context from Qdrant
   - Uses advanced reasoning to formulate response
   - Cites sources from documentation

5. **User receives answer** with sources and citations

## Benefits of This Hybrid Approach

✅ **No re-embedding needed** - Works with your existing Qdrant collection
✅ **Better responses** - GPT-4o-mini excels at technical explanations
✅ **Cost effective** - Embeddings are cheap/free with Gemini
✅ **Fast** - OpenAI responses are low-latency
✅ **Best of both worlds** - Gemini search + OpenAI intelligence

## Cost Breakdown

### Per 1000 Queries:
- **Gemini embeddings**: Free tier or ~$0.01
- **OpenAI GPT-4o-mini**: ~$1-5 depending on response length
- **Total**: ~$1-5 per 1000 queries (very affordable!)

### Compare to Full OpenAI:
- Full OpenAI (embeddings + LLM): ~$2-8 per 1000 queries
- Hybrid approach: ~$1-5 per 1000 queries
- **Savings**: 20-40% on embedding costs

## Troubleshooting

### "OPENAI_API_KEY not found"
- Check `.env` file exists in `mspbackups_chatbot/` directory
- Verify key starts with `sk-`
- No quotes around the key value

### "GEMINI_API_KEY not found"
- Check `.env` file has both keys
- Get key from https://makersuite.google.com/app/apikey
- No quotes around the key value

### "Dimension mismatch"
This shouldn't happen with correct config, but if it does:
- Verify `EMBEDDING_PROVIDER=gemini` (not openai)
- Check your Qdrant collection was created with Gemini (768d)

### Agent initialization fails
```bash
# Verify both packages installed
pip install openai google-generativeai

# Test imports
python -c "from openai import OpenAI; print('✓ OpenAI OK')"
python -c "import google.generativeai; print('✓ Gemini OK')"
```

## Alternative Configurations

### Option 1: Full OpenAI (requires re-embedding)
If you want to use OpenAI for both:

```bash
OPENAI_API_KEY=your_key
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

**⚠️ Requires:** Re-creating Qdrant collection with OpenAI embeddings (1536d)

### Option 2: Full Gemini (original setup)
If you want to use Gemini for both:

```bash
GEMINI_API_KEY=your_key
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
```

**✅ Works with:** Your existing Qdrant collection

### Option 3: Hybrid (current - recommended)
Best performance without re-embedding:

```bash
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=gemini
```

**✅ Works with:** Your existing Qdrant collection
**✅ Benefits:** Better responses, no data migration

## Performance Comparison

| Configuration | Response Quality | Search Quality | Cost | Re-embed Needed? |
|--------------|------------------|----------------|------|------------------|
| **Hybrid (OpenAI + Gemini)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $$ | ❌ No |
| Full OpenAI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ | ✅ Yes |
| Full Gemini | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $ | ❌ No |

## Testing

```bash
# Test with sample questions
python cli.py

# Try:
# - How to fix error 1531?
# - What is synthetic full backup?
# - Explain Forever Forward Incremental
```

## Summary

Your chatbot is configured to:
- ✅ Use **OpenAI GPT-4o-mini** for smart, fast responses
- ✅ Use **Gemini embeddings** to search your existing Qdrant collection
- ✅ Work immediately without re-embedding your data
- ✅ Provide excellent answers with source citations

**You get the best of both worlds!** 🎉

