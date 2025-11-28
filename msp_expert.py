"""
MSP360 Backup Expert AI Agent
Pydantic AI agent with tools for answering MSP360 Backup questions using RAG
Enhanced with query expansion and multi-query retrieval
"""

import os
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel

from embedding_service import EmbeddingService
from qdrant_tools import QdrantTools, format_search_results

load_dotenv()


@dataclass
class MSPDeps:
    """Dependencies for the MSP360 Backup expert agent"""
    qdrant_tools: QdrantTools
    embedding_service: EmbeddingService


# System prompt for the MSP360 Backup expert
SYSTEM_PROMPT = """
You are an MSP360 Backup expert assistant with access to the complete MSP360 knowledge base documentation from help.msp360.com and kb.msp360.com.

Your capabilities:
1. Answer questions about MSP360 Backup errors, warnings, and issues
2. Provide troubleshooting steps and solutions
3. Explain backup concepts (Synthetic Full, Forever Forward Incremental, VSS, GFS, etc.)
4. Help with configuration and best practices
5. Search by error codes and provide specific solutions
6. Cite specific documentation pages as sources

CRITICAL RULE - YOU MUST FOLLOW THIS:
🔴 NEVER answer from general knowledge alone
🔴 ALWAYS use retrieve_relevant_docs tool FIRST before answering ANY question
🔴 Base your answer ONLY on the documentation returned by the tools
🔴 If no documentation is found, say "I couldn't find specific documentation on this topic"
🔴 ALWAYS include the source URLs from the documentation in your response

Step-by-step process for EVERY question:
1. ✅ MANDATORY: Call retrieve_relevant_docs(query="user's question") 
2. ✅ Wait for the documentation results
3. ✅ Read the documentation content carefully
4. ✅ Formulate your answer based ONLY on the retrieved documentation
5. ✅ Include ALL source URLs at the end under "Sources:" section

Response Format:
**Overview**
Brief summary of the topic

**Details**
Detailed explanation from the documentation

**Key Points**
- Bullet points of important information
- Configuration steps if applicable
- Best practices if mentioned

**Sources**
- [Document Title](URL)
- [Another Document](URL)

CRITICAL FORMATTING RULES:
❌ Do NOT add any text before the Sources section like "For further reading...", "You can refer to...", "For more information..."
❌ Do NOT duplicate source listings
✅ Go directly from your answer to "**Sources**" with no transition text
✅ List each source as: - [Title](URL)
✅ Keep it clean and simple

CORRECT Example:
...end of your answer with key points.

**Sources**
- [About GFS](https://help.mspbackups.com/backup/about/gfs/gfs)
- [GFS Policy FAQ](https://help.mspbackups.com/...)

WRONG Example (DO NOT DO THIS):
...end of your answer.

For further reading, you can refer to the documentation using the following link:
**Sources**
- About GFS

Guidelines:
- If user mentions an error code (e.g., "code 1531"), use search_by_error_code for precise results
- Provide step-by-step solutions when troubleshooting
- Explain technical concepts clearly using the documentation
- Include code examples or configuration snippets from docs when relevant

Available documentation categories:
- backup/about: Core backup concepts and strategies
- backup/errors: Specific error codes and solutions
- backup/warnings: Warning messages
- restore: Restore procedures
- cloud-vendors: Cloud storage configurations

REMEMBER: Always use retrieve_relevant_docs FIRST, then answer based on the results!
"""

# Initialize LLM model
llm_provider = os.getenv("LLM_PROVIDER", "gemini")
llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

if llm_provider.lower() == "openai":
    from pydantic_ai.models.openai import OpenAIModel
    model = OpenAIModel(llm_model)
elif llm_provider.lower() == "anthropic":
    from pydantic_ai.models.anthropic import AnthropicModel
    model = AnthropicModel(llm_model)
elif llm_provider.lower() == "groq":
    from pydantic_ai.models.groq import GroqModel
    model = GroqModel(llm_model)
else:
    model = GeminiModel(llm_model)

# Create the agent
msp_expert = Agent(
    model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=MSPDeps,
    retries=2
)


@msp_expert.tool
async def retrieve_relevant_docs(
    ctx: RunContext[MSPDeps],
    query: str,
    category: Optional[str] = None,
    limit: int = 5
) -> str:
    """
    Search MSP360 Backup documentation using semantic similarity.
    ⚠️ USE THIS TOOL FIRST before answering ANY user question!
    
    This is your PRIMARY and REQUIRED tool for finding information.
    Do NOT answer questions without calling this tool first.
    
    Args:
        ctx: Run context with dependencies
        query: User's question or search term (be specific and detailed)
        category: Filter by category (backup/restore/errors/warnings) - optional
        limit: Number of results to return (default 5, max 10)
    
    Returns:
        Formatted documentation with titles, URLs, categories, error codes (if applicable), and full content.
        Base your answer ONLY on this returned content and cite the URLs provided.
    """
    try:
        # Limit the number of results
        limit = min(limit, 10)
        
        # Search the documentation
        results = await ctx.deps.qdrant_tools.search_docs(
            query=query,
            category=category,
            limit=limit
        )
        
        if not results:
            return f"No relevant documentation found for query: '{query}'"
        
        # Format results for LLM
        formatted = format_search_results(results)
        return formatted
        
    except Exception as e:
        return f"Error searching documentation: {str(e)}"


@msp_expert.tool
async def search_by_error_code(
    ctx: RunContext[MSPDeps],
    error_code: str
) -> str:
    """
    Search for documentation by specific error code.
    Use this when the user mentions a specific error code (e.g., "1531", "code 1531", "error 1531").
    
    Args:
        ctx: Run context with dependencies
        error_code: Error code to search for (e.g., "1531")
    
    Returns:
        Formatted documentation for that error code
    """
    try:
        # Clean error code (remove 'code' or 'error' prefix if present)
        clean_code = error_code.replace("code", "").replace("error", "").strip()
        
        # Search by error code
        results = await ctx.deps.qdrant_tools.search_by_error_code(clean_code)
        
        if not results:
            return f"No documentation found for error code: {error_code}"
        
        # Format results for LLM
        formatted = format_search_results(results)
        return formatted
        
    except Exception as e:
        return f"Error searching by error code: {str(e)}"


@msp_expert.tool
async def list_documentation_pages(
    ctx: RunContext[MSPDeps],
    category: Optional[str] = None
) -> str:
    """
    List available MSP360 Backup documentation pages.
    Use this when the user wants to browse or see what documentation is available.
    
    Args:
        ctx: Run context with dependencies
        category: Filter by category (Backup/Restore/Errors/Warnings/Cloud Vendors) - optional
    
    Returns:
        List of documentation page URLs with titles, optionally filtered by category
    """
    try:
        # Get list of pages
        pages = await ctx.deps.qdrant_tools.list_pages(category=category)
        
        if not pages:
            return "No pages found."
        
        # Format response
        category_str = f" in category '{category}'" if category else ""
        result = f"Found {len(pages)} documentation pages{category_str}:\n\n"
        
        # Show up to 50 pages
        result += "\n".join([f"- {page}" for page in pages[:50]])
        
        if len(pages) > 50:
            result += f"\n\n... and {len(pages) - 50} more pages"
        
        return result
        
    except Exception as e:
        return f"Error listing pages: {str(e)}"


@msp_expert.tool
async def get_page_content(
    ctx: RunContext[MSPDeps],
    url: str
) -> str:
    """
    Retrieve the full content of a specific documentation page by URL.
    Use this when the user asks for more details about a specific page or URL.
    
    Args:
        ctx: Run context with dependencies
        url: The MSP360 documentation URL (e.g., https://kb.msp360.com/backup/errors/synthetic-backup-not-supported)
    
    Returns:
        Complete page content
    """
    try:
        # Get the page
        page = await ctx.deps.qdrant_tools.get_page_by_url(url)
        
        if not page:
            return f"No content found for URL: {url}"
        
        # Format the page content
        error_code_str = ""
        if page.get("error_code"):
            error_code_str = f"\n**Error Code:** {page.get('error_code')}"
        
        source_str = f"\n**Source:** {page.get('source', 'Docs')}" if page.get('source') else ""
        
        result = f"""
# {page.get('title', 'Unknown')}

**URL:** {url}
**Category:** {page.get('category', 'Unknown')}{error_code_str}{source_str}

---

{page.get('text', 'No content available')}
"""
        return result
        
    except Exception as e:
        return f"Error retrieving page content: {str(e)}"


@msp_expert.tool
async def get_available_categories(ctx: RunContext[MSPDeps]) -> str:
    """
    Get list of all available documentation categories.
    Use this when the user wants to know what categories are available.
    
    Args:
        ctx: Run context with dependencies
    
    Returns:
        List of available categories with descriptions
    """
    try:
        categories = await ctx.deps.qdrant_tools.get_categories()
        
        category_descriptions = {
            "Backup": "Backup-related documentation and troubleshooting",
            "Restore": "Restore procedures and issues",
            "Errors": "Specific error codes and solutions",
            "Warnings": "Warning messages and their meanings",
            "Info Messages": "Informational messages",
            "Cloud Vendors": "Cloud storage provider configurations",
            "Best Practices": "Backup strategy recommendations",
            "Managed Backup Service": "MSP360 Managed Backup Service documentation",
            "Troubleshooting": "General troubleshooting guides"
        }
        
        result = "Available MSP360 Backup Documentation Categories:\n\n"
        for cat in categories:
            desc = category_descriptions.get(cat, "Documentation category")
            result += f"- **{cat}**: {desc}\n"
        
        return result
        
    except Exception as e:
        return f"Error getting categories: {str(e)}"


@msp_expert.tool
async def search_related_topics(
    ctx: RunContext[MSPDeps],
    topic: str,
    limit: int = 3
) -> str:
    """
    Find documentation related to a specific topic.
    Useful for suggesting related reading or exploring a topic area.
    
    Args:
        ctx: Run context with dependencies
        topic: The topic to find related documentation for
        limit: Number of related documents to return
    
    Returns:
        List of related documentation pages
    """
    try:
        # Search with topic-focused query
        results = await ctx.deps.qdrant_tools.search_docs(
            query=f"{topic} MSP360 backup documentation",
            limit=limit
        )
        
        if not results:
            return f"No related documentation found for topic: '{topic}'"
        
        # Format as a simple list
        related = f"Related documentation for '{topic}':\n\n"
        for r in results:
            related += f"- **{r.get('title', 'Unknown')}** ({r.get('category', 'Unknown')})\n"
            related += f"  {r.get('url', '')}\n"
        
        return related
        
    except Exception as e:
        return f"Error finding related topics: {str(e)}"


# Convenience function to create agent with dependencies
def create_msp_expert(
    qdrant_url: str = None,
    gemini_api_key: str = None,
    use_cache: bool = True
) -> tuple[Agent, MSPDeps]:
    """
    Create an MSP360 Backup expert agent with all dependencies.
    
    Args:
        qdrant_url: Qdrant connection URL (defaults to env var)
        gemini_api_key: Gemini API key (defaults to env var)
        use_cache: Whether to enable caching (default True)
    
    Returns:
        Tuple of (agent, dependencies)
    """
    embedding_service = EmbeddingService(api_key=gemini_api_key, use_cache=use_cache)
    qdrant_tools = QdrantTools(
        url=qdrant_url,
        embedding_service=embedding_service,
        use_cache=use_cache
    )
    
    deps = MSPDeps(
        qdrant_tools=qdrant_tools,
        embedding_service=embedding_service
    )
    
    return msp_expert, deps


# Example usage
async def ask_msp_question(question: str) -> str:
    """
    Ask the MSP360 Backup expert a question.
    
    Args:
        question: User's question
    
    Returns:
        Agent's response
    """
    agent, deps = create_msp_expert()
    result = await agent.run(question, deps=deps)
    return result.data
