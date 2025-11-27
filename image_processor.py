"""
Image Processing Service for MSP360 RAG Chatbot
Handles image uploads, OCR, and vision model integration for error screenshots
"""

import os
import base64
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ImageAnalysisResult:
    """Result from image analysis"""
    extracted_text: str
    error_codes: List[str]
    error_messages: List[str]
    suggested_query: str
    confidence: float
    raw_response: str


class ImageProcessor:
    """
    Process images (screenshots of errors) for the chatbot.
    
    Uses vision models to:
    1. Extract text from screenshots
    2. Identify error codes and messages
    3. Generate search queries for the documentation
    """
    
    def __init__(self, provider: str = None, api_key: str = None):
        """
        Initialize the image processor.
        
        Args:
            provider: Vision model provider (openai, gemini)
            api_key: API key for the provider
        """
        self.provider = (provider or os.getenv("VISION_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))).lower()
        
        if self.provider == "openai":
            from openai import AsyncOpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI vision")
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.model = os.getenv("VISION_MODEL", "gpt-4o")
        elif self.provider == "gemini":
            import google.generativeai as genai
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY required for Gemini vision")
            genai.configure(api_key=self.api_key)
            self.model = os.getenv("VISION_MODEL", "gemini-1.5-flash")
            self.genai = genai
        else:
            raise ValueError(f"Unsupported vision provider: {self.provider}")
    
    def _encode_image_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _detect_image_type(self, image_bytes: bytes) -> str:
        """Detect image MIME type from bytes"""
        # Check magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "image/webp"
        else:
            return "image/png"  # Default to PNG
    
    def _extract_error_codes(self, text: str) -> List[str]:
        """Extract potential error codes from text"""
        # Common patterns for error codes
        patterns = [
            r'\b(?:error\s*(?:code)?[:\s]*)?(\d{3,5})\b',  # Error 1531, code 1531
            r'\b0x[0-9A-Fa-f]{4,8}\b',  # Hex codes like 0x80070005
            r'\bE\d{4,6}\b',  # E-codes like E12345
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            codes.extend(matches)
        
        return list(set(codes))
    
    def _extract_error_messages(self, text: str) -> List[str]:
        """Extract potential error messages from text"""
        # Look for common error message patterns
        patterns = [
            r'(?:error|failed|failure|exception|warning)[:.]?\s*([^\n.]{10,100})',
            r'(?:cannot|could not|unable to)\s+([^\n.]{10,100})',
            r'(?:access denied|permission denied)([^\n.]{0,50})',
        ]
        
        messages = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            messages.extend([m.strip() for m in matches if m.strip()])
        
        return list(set(messages))[:5]  # Limit to top 5
    
    async def analyze_image(
        self,
        image_bytes: bytes,
        context: Optional[str] = None
    ) -> ImageAnalysisResult:
        """
        Analyze an image (screenshot) to extract error information.
        
        Args:
            image_bytes: Raw image bytes
            context: Optional context from user (e.g., "I got this error during backup")
        
        Returns:
            ImageAnalysisResult with extracted information
        """
        image_base64 = self._encode_image_base64(image_bytes)
        image_type = self._detect_image_type(image_bytes)
        
        prompt = """Analyze this screenshot from MSP360 Backup software.

Extract the following information:
1. All visible error codes (numeric codes like 1531, hex codes like 0x80070005)
2. Error messages or warning text
3. Any relevant technical details (paths, file names, service names)
4. The type of operation shown (backup, restore, configuration, etc.)

Format your response as:
ERROR_CODES: [list any error codes found, or "none"]
ERROR_MESSAGES: [key error/warning messages]
CONTEXT: [what operation or screen is shown]
SUGGESTED_QUERY: [a search query to find help for this issue]

Be precise and extract exact text from the image."""

        if context:
            prompt += f"\n\nAdditional context from user: {context}"
        
        try:
            if self.provider == "openai":
                response = await self._analyze_with_openai(image_base64, image_type, prompt)
            else:
                response = await self._analyze_with_gemini(image_bytes, prompt)
            
            return self._parse_analysis_response(response)
            
        except Exception as e:
            return ImageAnalysisResult(
                extracted_text="",
                error_codes=[],
                error_messages=[f"Failed to analyze image: {str(e)}"],
                suggested_query="",
                confidence=0.0,
                raw_response=str(e)
            )
    
    async def _analyze_with_openai(
        self,
        image_base64: str,
        image_type: str,
        prompt: str
    ) -> str:
        """Analyze image using OpenAI vision"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    async def _analyze_with_gemini(
        self,
        image_bytes: bytes,
        prompt: str
    ) -> str:
        """Analyze image using Gemini vision"""
        import asyncio
        from functools import partial
        
        model = self.genai.GenerativeModel(self.model)
        
        # Create image part
        image_part = {
            "mime_type": self._detect_image_type(image_bytes),
            "data": image_bytes
        }
        
        # Run synchronous API in executor
        loop = asyncio.get_event_loop()
        func = partial(
            model.generate_content,
            [prompt, image_part]
        )
        response = await loop.run_in_executor(None, func)
        
        return response.text
    
    def _parse_analysis_response(self, response: str) -> ImageAnalysisResult:
        """Parse the structured response from vision model"""
        error_codes = []
        error_messages = []
        suggested_query = ""
        
        # Parse ERROR_CODES
        codes_match = re.search(r'ERROR_CODES:\s*\[?([^\]\n]+)\]?', response, re.IGNORECASE)
        if codes_match:
            codes_text = codes_match.group(1)
            if 'none' not in codes_text.lower():
                # Extract all numeric/hex codes
                error_codes = self._extract_error_codes(codes_text)
        
        # Parse ERROR_MESSAGES
        messages_match = re.search(r'ERROR_MESSAGES:\s*\[?([^\]\n]+)\]?', response, re.IGNORECASE)
        if messages_match:
            messages_text = messages_match.group(1)
            if 'none' not in messages_text.lower():
                error_messages = [m.strip() for m in messages_text.split(',') if m.strip()]
        
        # Parse SUGGESTED_QUERY
        query_match = re.search(r'SUGGESTED_QUERY:\s*\[?([^\]\n]+)\]?', response, re.IGNORECASE)
        if query_match:
            suggested_query = query_match.group(1).strip()
        
        # Calculate confidence based on what was found
        confidence = 0.0
        if error_codes:
            confidence += 0.4
        if error_messages:
            confidence += 0.3
        if suggested_query:
            confidence += 0.3
        
        return ImageAnalysisResult(
            extracted_text=response,
            error_codes=error_codes,
            error_messages=error_messages[:5],
            suggested_query=suggested_query,
            confidence=confidence,
            raw_response=response
        )
    
    async def process_and_search(
        self,
        image_bytes: bytes,
        qdrant_tools,
        context: Optional[str] = None
    ) -> Tuple[ImageAnalysisResult, List[Dict[str, Any]]]:
        """
        Analyze image and search documentation for solutions.
        
        Args:
            image_bytes: Raw image bytes
            qdrant_tools: QdrantTools instance for searching
            context: Optional user context
        
        Returns:
            Tuple of (analysis result, search results)
        """
        # Analyze the image
        analysis = await self.analyze_image(image_bytes, context)
        
        search_results = []
        
        # Search by error codes first
        for code in analysis.error_codes:
            results = await qdrant_tools.search_by_error_code(code)
            search_results.extend(results)
        
        # Search by suggested query
        if analysis.suggested_query:
            results = await qdrant_tools.search_docs(
                query=analysis.suggested_query,
                limit=5
            )
            search_results.extend(results)
        
        # Search by error messages
        for msg in analysis.error_messages[:2]:  # Limit to first 2 messages
            results = await qdrant_tools.search_docs(
                query=msg,
                limit=3
            )
            search_results.extend(results)
        
        # Deduplicate results by URL
        seen_urls = set()
        unique_results = []
        for r in search_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)
        
        return analysis, unique_results[:10]


def format_image_analysis_for_agent(
    analysis: ImageAnalysisResult,
    search_results: List[Dict[str, Any]]
) -> str:
    """
    Format image analysis and search results for the agent.
    
    Args:
        analysis: Image analysis result
        search_results: Search results from documentation
    
    Returns:
        Formatted string for the agent
    """
    output = "## Image Analysis Results\n\n"
    
    if analysis.error_codes:
        output += f"**Error Codes Found:** {', '.join(analysis.error_codes)}\n\n"
    
    if analysis.error_messages:
        output += "**Error Messages:**\n"
        for msg in analysis.error_messages:
            output += f"- {msg}\n"
        output += "\n"
    
    if analysis.suggested_query:
        output += f"**Suggested Query:** {analysis.suggested_query}\n\n"
    
    output += f"**Analysis Confidence:** {analysis.confidence:.0%}\n\n"
    
    output += "---\n\n"
    
    if search_results:
        output += "## Related Documentation\n\n"
        for i, result in enumerate(search_results[:5], 1):
            output += f"### {i}. {result.get('title', 'Unknown')}\n"
            output += f"**URL:** {result.get('url', 'Unknown')}\n"
            if result.get('error_code'):
                output += f"**Error Code:** {result.get('error_code')}\n"
            output += f"\n{result.get('text', '')[:500]}...\n\n"
            output += "---\n\n"
    else:
        output += "No related documentation found for this error.\n"
    
    return output

