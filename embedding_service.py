"""
Embedding Service
Wrapper for generating embeddings using OpenAI or Gemini API
"""

import os
import asyncio
from typing import List
from functools import partial
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """Service for generating text embeddings using OpenAI or Gemini"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        provider: str = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: API key (defaults to env var based on provider)
            model: Embedding model name (defaults based on provider)
            provider: Embedding provider (openai or gemini, defaults to openai)
        """
        self.provider = (provider or os.getenv("EMBEDDING_PROVIDER", "openai")).lower()
        
        if self.provider == "openai":
            from openai import AsyncOpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.dimension = 1536 if "3-small" in self.model else 3072
        else:
            # Gemini
            import google.generativeai as genai
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            model_name = model or os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
            
            # Ensure model name has correct prefix
            if not model_name.startswith(("models/", "tunedModels/")):
                model_name = f"models/{model_name}"
            
            self.model = model_name
            
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            
            genai.configure(api_key=self.api_key)
            self.dimension = 768
    
    async def generate_embedding(
        self,
        text: str,
        task_type: str = "retrieval_query"
    ) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            task_type: Task type for embedding (retrieval_query or retrieval_document) - used for Gemini
        
        Returns:
            Embedding vector
        """
        try:
            if self.provider == "openai":
                # OpenAI embeddings
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            else:
                # Gemini embeddings
                import google.generativeai as genai
                loop = asyncio.get_event_loop()
                func = partial(
                    genai.embed_content,
                    model=self.model,
                    content=text,
                    task_type=task_type
                )
                result = await loop.run_in_executor(None, func)
                return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embedding_sync(
        self,
        text: str,
        task_type: str = "retrieval_query"
    ) -> List[float]:
        """
        Synchronous version of generate_embedding.
        
        Args:
            text: Text to embed
            task_type: Task type for embedding (for Gemini)
        
        Returns:
            Embedding vector
        """
        try:
            if self.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            else:
                import google.generativeai as genai
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type=task_type
                )
                return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "retrieval_query"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            task_type: Task type for embeddings
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text, task_type)
            embeddings.append(embedding)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.dimension


# Convenience function for quick usage
async def get_embedding(text: str, task_type: str = "retrieval_query") -> List[float]:
    """
    Quick function to get embedding for text.
    
    Args:
        text: Text to embed
        task_type: Task type for embedding
    
    Returns:
        Embedding vector
    """
    service = EmbeddingService()
    return await service.generate_embedding(text, task_type)

