"""
Analytics Service for MSP360 RAG Chatbot
Tracks queries, response times, feedback, and usage patterns
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryRecord:
    """Record of a single query"""
    query_id: str
    query: str
    response_time: float
    timestamp: str
    category: Optional[str] = None
    error_code: Optional[str] = None
    sources_used: List[str] = field(default_factory=list)
    feedback: Optional[str] = None
    feedback_comment: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class AnalyticsSummary:
    """Summary of analytics data"""
    total_queries: int
    avg_response_time: float
    positive_feedback_rate: float
    negative_feedback_rate: float
    top_queries: List[Dict[str, Any]]
    top_error_codes: List[Dict[str, Any]]
    queries_by_hour: Dict[int, int]
    queries_by_day: Dict[str, int]


class AnalyticsService:
    """
    Service for tracking and analyzing chatbot usage.
    
    In production, this would use a proper database (PostgreSQL, etc.).
    This implementation uses in-memory storage with optional file persistence.
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_records: int = 10000
    ):
        """
        Initialize analytics service.
        
        Args:
            persist_path: Optional path to persist data (JSON file)
            max_records: Maximum records to keep in memory
        """
        self.persist_path = persist_path or os.getenv("ANALYTICS_PERSIST_PATH")
        self.max_records = max_records
        
        self._records: List[QueryRecord] = []
        self._lock = threading.RLock()
        
        # Load existing data if persist path exists
        if self.persist_path and Path(self.persist_path).exists():
            self._load_from_file()
    
    def _generate_id(self, query: str) -> str:
        """Generate unique query ID"""
        timestamp = str(time.time())
        return hashlib.md5(f"{query}{timestamp}".encode()).hexdigest()[:12]
    
    def _load_from_file(self):
        """Load records from persistence file"""
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
                self._records = [
                    QueryRecord(**r) for r in data.get("records", [])
                ]
        except Exception as e:
            print(f"Error loading analytics data: {e}")
    
    def _save_to_file(self):
        """Save records to persistence file"""
        if not self.persist_path:
            return
        
        try:
            data = {
                "records": [asdict(r) for r in self._records],
                "last_updated": datetime.now().isoformat()
            }
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving analytics data: {e}")
    
    def record_query(
        self,
        query: str,
        response_time: float,
        category: Optional[str] = None,
        error_code: Optional[str] = None,
        sources_used: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Record a query.
        
        Args:
            query: The user's query
            response_time: Response time in seconds
            category: Category filter used
            error_code: Error code if searched
            sources_used: List of source URLs used
            user_id: Optional user identifier
        
        Returns:
            Query ID for feedback tracking
        """
        query_id = self._generate_id(query)
        
        record = QueryRecord(
            query_id=query_id,
            query=query,
            response_time=response_time,
            timestamp=datetime.now().isoformat(),
            category=category,
            error_code=error_code,
            sources_used=sources_used or [],
            user_id=user_id
        )
        
        with self._lock:
            self._records.append(record)
            
            # Trim if over limit
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]
            
            # Persist periodically
            if len(self._records) % 100 == 0:
                self._save_to_file()
        
        return query_id
    
    def record_feedback(
        self,
        query_id: str,
        rating: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Record feedback for a query.
        
        Args:
            query_id: Query ID from record_query
            rating: "positive" or "negative"
            comment: Optional feedback comment
        
        Returns:
            True if feedback was recorded
        """
        with self._lock:
            for record in reversed(self._records):
                if record.query_id == query_id:
                    record.feedback = rating
                    record.feedback_comment = comment
                    self._save_to_file()
                    return True
        return False
    
    def get_summary(
        self,
        days: int = 7,
        top_n: int = 10
    ) -> AnalyticsSummary:
        """
        Get analytics summary.
        
        Args:
            days: Number of days to include
            top_n: Number of top items to return
        
        Returns:
            AnalyticsSummary with aggregated data
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._lock:
            # Filter records by date
            recent_records = [
                r for r in self._records
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]
        
        if not recent_records:
            return AnalyticsSummary(
                total_queries=0,
                avg_response_time=0.0,
                positive_feedback_rate=0.0,
                negative_feedback_rate=0.0,
                top_queries=[],
                top_error_codes=[],
                queries_by_hour={},
                queries_by_day={}
            )
        
        # Calculate metrics
        total = len(recent_records)
        avg_time = sum(r.response_time for r in recent_records) / total
        
        # Feedback rates
        with_feedback = [r for r in recent_records if r.feedback]
        positive = sum(1 for r in with_feedback if r.feedback == "positive")
        negative = sum(1 for r in with_feedback if r.feedback == "negative")
        feedback_total = len(with_feedback)
        
        positive_rate = positive / feedback_total if feedback_total > 0 else 0.0
        negative_rate = negative / feedback_total if feedback_total > 0 else 0.0
        
        # Top queries (by frequency)
        query_counts = defaultdict(int)
        for r in recent_records:
            # Normalize query for grouping
            normalized = r.query.lower().strip()[:100]
            query_counts[normalized] += 1
        
        top_queries = [
            {"query": q, "count": c}
            for q, c in sorted(query_counts.items(), key=lambda x: -x[1])[:top_n]
        ]
        
        # Top error codes
        error_counts = defaultdict(int)
        for r in recent_records:
            if r.error_code:
                error_counts[r.error_code] += 1
        
        top_error_codes = [
            {"error_code": e, "count": c}
            for e, c in sorted(error_counts.items(), key=lambda x: -x[1])[:top_n]
        ]
        
        # Queries by hour
        queries_by_hour = defaultdict(int)
        for r in recent_records:
            hour = datetime.fromisoformat(r.timestamp).hour
            queries_by_hour[hour] += 1
        
        # Queries by day
        queries_by_day = defaultdict(int)
        for r in recent_records:
            day = datetime.fromisoformat(r.timestamp).strftime("%Y-%m-%d")
            queries_by_day[day] += 1
        
        return AnalyticsSummary(
            total_queries=total,
            avg_response_time=round(avg_time, 3),
            positive_feedback_rate=round(positive_rate, 3),
            negative_feedback_rate=round(negative_rate, 3),
            top_queries=top_queries,
            top_error_codes=top_error_codes,
            queries_by_hour=dict(queries_by_hour),
            queries_by_day=dict(queries_by_day)
        )
    
    def get_unanswered_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get queries with negative feedback or no sources.
        These represent potential knowledge gaps.
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of problematic queries
        """
        with self._lock:
            problematic = [
                r for r in self._records
                if r.feedback == "negative" or not r.sources_used
            ]
        
        # Sort by recency
        problematic.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "query": r.query,
                "timestamp": r.timestamp,
                "feedback": r.feedback,
                "feedback_comment": r.feedback_comment,
                "sources_found": len(r.sources_used)
            }
            for r in problematic[:limit]
        ]
    
    def export_data(self, format: str = "json") -> str:
        """
        Export all analytics data.
        
        Args:
            format: Export format ("json" or "csv")
        
        Returns:
            Exported data as string
        """
        with self._lock:
            records = [asdict(r) for r in self._records]
        
        if format == "json":
            return json.dumps(records, indent=2)
        elif format == "csv":
            if not records:
                return ""
            
            headers = records[0].keys()
            lines = [",".join(headers)]
            
            for r in records:
                values = [str(r.get(h, "")).replace(",", ";") for h in headers]
                lines.append(",".join(values))
            
            return "\n".join(lines)
        
        raise ValueError(f"Unsupported format: {format}")
    
    def clear(self):
        """Clear all analytics data"""
        with self._lock:
            self._records = []
            self._save_to_file()


# Global analytics instance
_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Get or create global analytics service"""
    global _analytics_service
    if _analytics_service is None:
        persist_path = os.getenv("ANALYTICS_PERSIST_PATH", "analytics_data.json")
        _analytics_service = AnalyticsService(persist_path=persist_path)
    return _analytics_service

