"""
Analytics Dashboard for MSP360 Backup Expert
Streamlit page for viewing usage analytics and insights
"""

import streamlit as st
from datetime import datetime, timedelta
import json

from analytics import get_analytics_service, AnalyticsSummary
from cache_service import get_all_cache_stats


# Page configuration
st.set_page_config(
    page_title="Analytics Dashboard - MSP360 Expert",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #1f77b4;
}
.metric-label {
    color: #666;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Analytics Dashboard")
st.markdown("Monitor usage patterns, performance metrics, and identify knowledge gaps.")

# Sidebar controls
with st.sidebar:
    st.header("Dashboard Controls")
    
    days_range = st.slider("Time Range (days)", 1, 30, 7)
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    st.markdown("---")
    
    # Export options
    st.subheader("Export Data")
    export_format = st.selectbox("Format", ["JSON", "CSV"])
    
    analytics = get_analytics_service()
    if st.button("📥 Download Export"):
        data = analytics.export_data(format=export_format.lower())
        st.download_button(
            label="Download",
            data=data,
            file_name=f"analytics_export_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}",
            mime="application/json" if export_format == "JSON" else "text/csv"
        )

# Get analytics data
analytics = get_analytics_service()
summary = analytics.get_summary(days=days_range)

# Key Metrics Row
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Queries",
        value=summary.total_queries,
        help="Total number of queries in the selected time period"
    )

with col2:
    st.metric(
        label="Avg Response Time",
        value=f"{summary.avg_response_time:.2f}s",
        help="Average time to generate a response"
    )

with col3:
    if summary.positive_feedback_rate > 0 or summary.negative_feedback_rate > 0:
        satisfaction = summary.positive_feedback_rate / (
            summary.positive_feedback_rate + summary.negative_feedback_rate
        ) * 100 if (summary.positive_feedback_rate + summary.negative_feedback_rate) > 0 else 0
        st.metric(
            label="Satisfaction Rate",
            value=f"{satisfaction:.0f}%",
            help="Percentage of positive feedback"
        )
    else:
        st.metric(
            label="Satisfaction Rate",
            value="N/A",
            help="No feedback data yet"
        )

with col4:
    cache_stats = get_all_cache_stats()
    embedding_hit_rate = cache_stats["embedding_cache"].get("hit_rate", "0%")
    st.metric(
        label="Cache Hit Rate",
        value=embedding_hit_rate,
        help="Embedding cache hit rate"
    )

st.markdown("---")

# Charts Row
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Queries Over Time")
    
    if summary.queries_by_day:
        # Sort by date
        sorted_days = sorted(summary.queries_by_day.items())
        
        chart_data = {
            "Date": [d[0] for d in sorted_days],
            "Queries": [d[1] for d in sorted_days]
        }
        
        st.line_chart(chart_data, x="Date", y="Queries")
    else:
        st.info("No query data available for the selected period.")

with col2:
    st.subheader("⏰ Queries by Hour")
    
    if summary.queries_by_hour:
        # Create full 24-hour data
        hourly_data = {
            "Hour": list(range(24)),
            "Queries": [summary.queries_by_hour.get(h, 0) for h in range(24)]
        }
        
        st.bar_chart(hourly_data, x="Hour", y="Queries")
    else:
        st.info("No hourly data available.")

st.markdown("---")

# Top Queries and Error Codes
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Top Queries")
    
    if summary.top_queries:
        for i, q in enumerate(summary.top_queries[:10], 1):
            with st.container():
                st.markdown(f"**{i}.** {q['query'][:80]}{'...' if len(q['query']) > 80 else ''}")
                st.caption(f"Count: {q['count']}")
    else:
        st.info("No queries recorded yet.")

with col2:
    st.subheader("🔢 Top Error Codes")
    
    if summary.top_error_codes:
        for i, e in enumerate(summary.top_error_codes[:10], 1):
            with st.container():
                st.markdown(f"**{i}. Error {e['error_code']}**")
                st.caption(f"Searched: {e['count']} times")
    else:
        st.info("No error code searches recorded.")

st.markdown("---")

# Knowledge Gaps
st.subheader("⚠️ Knowledge Gaps")
st.caption("Queries with negative feedback or no documentation found")

unanswered = analytics.get_unanswered_queries(limit=10)

if unanswered:
    for i, q in enumerate(unanswered, 1):
        with st.expander(f"{i}. {q['query'][:60]}{'...' if len(q['query']) > 60 else ''}"):
            st.markdown(f"**Query:** {q['query']}")
            st.markdown(f"**Time:** {q['timestamp']}")
            st.markdown(f"**Sources Found:** {q['sources_found']}")
            if q['feedback']:
                st.markdown(f"**Feedback:** {q['feedback']}")
            if q['feedback_comment']:
                st.markdown(f"**Comment:** {q['feedback_comment']}")
else:
    st.success("No knowledge gaps identified! All queries appear to be well-answered.")

st.markdown("---")

# Cache Statistics
st.subheader("💾 Cache Statistics")

cache_stats = get_all_cache_stats()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Embedding Cache**")
    embed_stats = cache_stats["embedding_cache"]
    st.json(embed_stats)

with col2:
    st.markdown("**Search Cache**")
    search_stats = cache_stats["search_cache"]
    st.json(search_stats)

# Admin Actions
st.markdown("---")
st.subheader("🛠️ Admin Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Analytics Data", type="secondary"):
        if st.session_state.get("confirm_clear_analytics"):
            analytics.clear()
            st.success("Analytics data cleared!")
            st.session_state["confirm_clear_analytics"] = False
            st.rerun()
        else:
            st.session_state["confirm_clear_analytics"] = True
            st.warning("Click again to confirm clearing all analytics data.")

with col2:
    from cache_service import clear_all_caches
    if st.button("🗑️ Clear All Caches", type="secondary"):
        if st.session_state.get("confirm_clear_cache"):
            clear_all_caches()
            st.success("All caches cleared!")
            st.session_state["confirm_clear_cache"] = False
            st.rerun()
        else:
            st.session_state["confirm_clear_cache"] = True
            st.warning("Click again to confirm clearing all caches.")

with col3:
    # Placeholder for future admin action
    st.button("📊 Generate Report", type="secondary", disabled=True)
    st.caption("Coming soon")

# Footer
st.markdown("---")
st.caption(f"Dashboard updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

