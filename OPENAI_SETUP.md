# OpenAI Configuration Guide

The MSP360 Backup Chatbot now uses **OpenAI** for both LLM (GPT-4o-mini) and embeddings (text-embedding-3-small).

## Quick Setup

### 1. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (you won't be able to see it again!)

### 2. Configure Environment

Create a `.env` file in the `mspbackups_chatbot` directory:

```bash
# Copy the template
cp .env.template .env

# Edit with your API key
nano .env
```

Add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=msp_docs_v2
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Install/Update Dependencies

```bash
# Activate venv if not already active
source venv/bin/activate

# Install/update dependencies (includes openai package)
pip install -r requirements.txt
```

### 4. Verify Setup

```bash
# Test import
python -c "from openai import OpenAI; print('✓ OpenAI installed')"

# Test the chatbot
python cli.py
```

## Model Options

### LLM Models (for chat responses)

- **gpt-4o-mini** (default) - Fast, cost-effective, intelligent
- **gpt-4o** - Most capable, higher cost
- **gpt-4-turbo** - Previous generation, still powerful
- **gpt-3.5-turbo** - Fastest, lowest cost

Update in `.env`:
```bash
LLM_MODEL=gpt-4o  # or gpt-4-turbo, gpt-3.5-turbo
```

### Embedding Models (for search)

- **text-embedding-3-small** (default) - 1536 dimensions, cost-effective
- **text-embedding-3-large** - 3072 dimensions, higher quality
- **text-embedding-ada-002** - Legacy model, still works

Update in `.env`:
```bash
EMBEDDING_MODEL=text-embedding-3-large  # for better search quality
```

## Cost Estimation

Approximate costs per 1000 queries:

### With gpt-4o-mini (default):
- **LLM**: ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
- **Embeddings**: ~$0.02 per 1M tokens
- **Typical cost per query**: $0.001-0.005 (0.1-0.5 cents)

### With gpt-4o:
- **LLM**: ~$2.50 per 1M input tokens, $10 per 1M output tokens
- **Embeddings**: same as above
- **Typical cost per query**: $0.01-0.03 (1-3 cents)

## Switching Back to Gemini

If you want to use Gemini instead, update `.env`:

```bash
GEMINI_API_KEY=your_gemini_key
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
```

## Hybrid Configuration

You can mix providers! For example, use OpenAI for LLM and Gemini for embeddings:

```bash
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
```

**Note**: If switching embedding providers, you'll need to **re-embed your Qdrant collection** as the vector dimensions differ:
- OpenAI: 1536 or 3072 dimensions
- Gemini: 768 dimensions

## Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure `.env` file exists in mspbackups_chatbot directory
- Verify the key is on the correct line without quotes
- Check for spaces: `OPENAI_API_KEY=sk-...` (no spaces around =)

### "Rate limit exceeded"
- OpenAI has rate limits on free tier
- Wait a minute and try again
- Consider upgrading to paid tier

### "Invalid API key"
- Verify key is correct and active at https://platform.openai.com/api-keys
- Key should start with `sk-proj-` or `sk-`

### Import errors
```bash
pip install --upgrade openai
```

## Benefits of OpenAI

✅ **Better reasoning** - GPT-4o-mini is excellent at technical documentation
✅ **Faster responses** - Lower latency than Gemini
✅ **Better structured output** - Consistent formatting
✅ **Function calling** - Works seamlessly with Pydantic AI tools
✅ **Wide availability** - Works in most regions

## Resources

- [OpenAI Platform](https://platform.openai.com/)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [OpenAI Documentation](https://platform.openai.com/docs/)
- [Usage Dashboard](https://platform.openai.com/usage)

