"""
Streamlit Chat Interface for MSP360 Backup Expert
Web UI with conversation history, streaming, feedback, and export features
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime
from typing import Optional
import streamlit as st
from dotenv import load_dotenv

from msp_expert import create_msp_expert

load_dotenv()


# Page configuration
st.set_page_config(
    page_title="MSP360 Backup Expert Assistant",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Version info for testing CI/CD pipeline
VERSION = "v2.0.0"

# Initialize theme in session state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS with dark mode support
def get_custom_css(dark_mode: bool = False) -> str:
    if dark_mode:
        return """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            background-color: #1a1a2e;
            color: #eaeaea;
        }
        .source-box {
            background-color: #16213e;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            border: 1px solid #0f3460;
        }
        .feedback-btn {
            display: inline-block;
            padding: 5px 15px;
            margin: 5px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .feedback-positive {
            background-color: #0f3460;
            border: 1px solid #00d9ff;
        }
        .feedback-positive:hover {
            background-color: #1a4a7a;
        }
        .feedback-negative {
            background-color: #0f3460;
            border: 1px solid #ff6b6b;
        }
        .feedback-negative:hover {
            background-color: #3a1a1a;
        }
        .suggestion-chip {
            display: inline-block;
            padding: 8px 16px;
            margin: 4px;
            background-color: #16213e;
            border: 1px solid #0f3460;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .suggestion-chip:hover {
            background-color: #1a4a7a;
            border-color: #00d9ff;
        }
        .stChatMessage {
            background-color: #16213e !important;
        }
        .stMarkdown a {
            color: #00d9ff !important;
        }
        div[data-testid="stSidebar"] {
            background-color: #16213e;
        }
        </style>
        """
    else:
        return """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .source-box {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .feedback-btn {
            display: inline-block;
            padding: 5px 15px;
            margin: 5px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .feedback-positive {
            background-color: #e8f5e9;
            border: 1px solid #4caf50;
        }
        .feedback-positive:hover {
            background-color: #c8e6c9;
        }
        .feedback-negative {
            background-color: #ffebee;
            border: 1px solid #f44336;
        }
        .feedback-negative:hover {
            background-color: #ffcdd2;
        }
        .suggestion-chip {
            display: inline-block;
            padding: 8px 16px;
            margin: 4px;
            background-color: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .suggestion-chip:hover {
            background-color: #bbdefb;
        }
        </style>
        """

st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # message_id -> {"rating": "positive"/"negative", "timestamp": ...}

if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "queries": [],
        "response_times": [],
        "error_codes_searched": [],
        "categories_accessed": []
    }

if "agent" not in st.session_state:
    try:
        st.session_state.agent, st.session_state.deps = create_msp_expert()
        st.session_state.agent_ready = True
    except Exception as e:
        st.session_state.agent_ready = False
        st.session_state.error = str(e)


def generate_message_id(content: str, timestamp: float) -> str:
    """Generate a unique message ID"""
    return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()[:12]


def export_chat_markdown() -> str:
    """Export chat history as Markdown"""
    export = f"# MSP360 Backup Expert Chat Export\n"
    export += f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    export += "---\n\n"
    
    for msg in st.session_state.messages:
        role = "**You:**" if msg["role"] == "user" else "**Assistant:**"
        export += f"{role}\n\n{msg['content']}\n\n---\n\n"
    
    return export


def record_analytics(query: str, response_time: float, error_code: Optional[str] = None):
    """Record query analytics"""
    st.session_state.analytics["queries"].append({
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "response_time": response_time
    })
    st.session_state.analytics["response_times"].append(response_time)
    
    if error_code:
        st.session_state.analytics["error_codes_searched"].append(error_code)


async def run_agent_streaming(prompt: str, message_placeholder):
    """Run agent with streaming response"""
    full_response = ""
    start_time = time.time()
    
    try:
        # Try streaming first
        async with st.session_state.agent.run_stream(
            prompt,
            deps=st.session_state.deps
        ) as result:
            async for text in result.stream_text(delta=True):
                full_response += text
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        response_time = time.time() - start_time
        record_analytics(prompt, response_time)
        return full_response
        
    except Exception as stream_error:
        # Fallback to non-streaming if streaming fails
        message_placeholder.markdown("🤔 Processing your question...")
        result = await st.session_state.agent.run(
            prompt,
            deps=st.session_state.deps
        )
        full_response = result.data
        message_placeholder.markdown(full_response)
        response_time = time.time() - start_time
        record_analytics(prompt, response_time)
        return full_response


# Sidebar
with st.sidebar:
    st.title("💾 MSP360 Backup Expert")
    st.caption(f"Version {VERSION}")
    st.markdown("---")
    
    # Theme toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("🌙 Dark Mode")
    with col2:
        if st.toggle("", value=st.session_state.dark_mode, key="theme_toggle"):
            st.session_state.dark_mode = True
            st.rerun()
        else:
            if st.session_state.dark_mode:
                st.session_state.dark_mode = False
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    ### About
    AI assistant with access to the complete MSP360 Backup knowledge base documentation.
    
    ### Capabilities
    - 🔧 Troubleshoot backup errors
    - 🔢 Explain error codes
    - 📋 Provide step-by-step solutions
    - ⚙️ Configuration guidance
    - ✅ Best practices advice
    - ☁️ Cloud storage setup help
    
    ### Documentation Sources
    - **help.msp360.com**: Official documentation
    - **kb.msp360.com**: Knowledge base articles
    - **Error codes**: Detailed error explanations
    - **Troubleshooting**: Step-by-step guides
    """)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback = {}
            st.rerun()
    
    with col2:
        # Export chat button
        if st.session_state.messages:
            export_data = export_chat_markdown()
            st.download_button(
                label="📥 Export",
                data=export_data,
                file_name=f"msp360_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Analytics summary
    with st.expander("📊 Session Analytics"):
        total_queries = len(st.session_state.analytics["queries"])
        if total_queries > 0:
            avg_time = sum(st.session_state.analytics["response_times"]) / total_queries
            st.metric("Total Queries", total_queries)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            positive = sum(1 for f in st.session_state.feedback.values() if f.get("rating") == "positive")
            negative = sum(1 for f in st.session_state.feedback.values() if f.get("rating") == "negative")
            if positive + negative > 0:
                st.metric("Feedback", f"👍 {positive} / 👎 {negative}")
        else:
            st.caption("No queries yet")
    
    st.markdown("---")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - How to fix error code 1531?
        - What is synthetic full backup?
        - How to configure Forever Forward Incremental?
        - VSS error access denied
        - How to restore a backup plan?
        - What cloud storage supports synthetic backup?
        - Troubleshoot CrashLoopBackOff error
        - How to fix I/O error code 1076?
        - Explain backup retention policy
        """)


# Main content
st.title("💾 MSP360 Backup Expert Assistant")
st.markdown("Ask questions about MSP360 Backup and get expert answers with documentation sources!")

# Check if agent is ready
if not st.session_state.agent_ready:
    st.error(f"""
    ❌ Failed to initialize agent: {st.session_state.get('error', 'Unknown error')}
    
    Please check:
    1. .env file exists with OPENAI_API_KEY and GEMINI_API_KEY
    2. LLM_PROVIDER=openai and EMBEDDING_PROVIDER=gemini
    3. Qdrant is running at QDRANT_URL
    4. msp360_docs collection exists in Qdrant
    """)
    st.stop()

# Display chat messages with feedback buttons
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add feedback buttons for assistant messages
        if message["role"] == "assistant" and not message["content"].startswith("❌"):
            msg_id = message.get("id", f"msg_{idx}")
            
            col1, col2, col3 = st.columns([1, 1, 10])
            
            current_feedback = st.session_state.feedback.get(msg_id, {}).get("rating")
            
            with col1:
                if st.button(
                    "👍" + (" ✓" if current_feedback == "positive" else ""),
                    key=f"pos_{msg_id}",
                    help="This answer was helpful"
                ):
                    st.session_state.feedback[msg_id] = {
                        "rating": "positive",
                        "timestamp": datetime.now().isoformat()
                    }
                    st.rerun()
            
            with col2:
                if st.button(
                    "👎" + (" ✓" if current_feedback == "negative" else ""),
                    key=f"neg_{msg_id}",
                    help="This answer needs improvement"
                ):
                    st.session_state.feedback[msg_id] = {
                        "rating": "negative",
                        "timestamp": datetime.now().isoformat()
                    }
                    st.rerun()

# Suggested follow-up questions (shown after responses)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_response = st.session_state.messages[-1]["content"]
    
    # Generate contextual suggestions based on content
    suggestions = []
    if "error" in last_response.lower() or "code" in last_response.lower():
        suggestions.append("What are other common error codes?")
    if "backup" in last_response.lower():
        suggestions.append("What are backup best practices?")
    if "restore" in last_response.lower():
        suggestions.append("How to verify restore integrity?")
    if "vss" in last_response.lower():
        suggestions.append("How to troubleshoot VSS issues?")
    if "synthetic" in last_response.lower():
        suggestions.append("What storage supports synthetic backups?")
    
    # Add generic suggestions if we don't have enough
    default_suggestions = [
        "Show me related documentation",
        "What are the prerequisites?",
        "Are there any known issues?"
    ]
    
    while len(suggestions) < 3 and default_suggestions:
        suggestions.append(default_suggestions.pop(0))
    
    if suggestions:
        st.markdown("##### 💡 Related Questions")
        cols = st.columns(len(suggestions[:3]))
        for i, suggestion in enumerate(suggestions[:3]):
            with cols[i]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    # Trigger the suggestion as a new query
                    st.session_state.pending_query = suggestion
                    st.rerun()

# Handle pending query from suggestions
if "pending_query" in st.session_state:
    prompt = st.session_state.pending_query
    del st.session_state.pending_query
    
    # Add user message
    msg_id = generate_message_id(prompt, time.time())
    st.session_state.messages.append({"role": "user", "content": prompt, "id": msg_id})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Searching documentation...")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                run_agent_streaming(prompt, message_placeholder)
            )
            loop.close()
            
            resp_id = generate_message_id(response, time.time())
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "id": resp_id
            })
            st.rerun()
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })

# Chat input
if prompt := st.chat_input("Ask about MSP360 Backup..."):
    # Add user message to chat
    msg_id = generate_message_id(prompt, time.time())
    st.session_state.messages.append({"role": "user", "content": prompt, "id": msg_id})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Searching documentation...")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                run_agent_streaming(prompt, message_placeholder)
            )
            loop.close()
            
            # Add assistant response to chat with ID for feedback
            resp_id = generate_message_id(response, time.time())
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "id": resp_id
            })
            
            # Rerun to show feedback buttons
            st.rerun()
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })

# Footer
st.markdown("---")
footer_style = "color: #888;" if st.session_state.dark_mode else "color: #666;"
st.markdown(f"""
<div style='text-align: center; {footer_style} font-size: 0.9em;'>
    Powered by Pydantic AI + OpenAI (LLM) + Gemini (Embeddings) + Qdrant | 
    <a href='https://help.msp360.com' target='_blank'>MSP360 Documentation</a> | 
    <a href='https://kb.msp360.com' target='_blank'>Knowledge Base</a>
</div>
""", unsafe_allow_html=True)
