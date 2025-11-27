"""
Streamlit Chat Interface for MSP360 Backup Expert
Web UI with conversation history
"""

import asyncio
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

# Custom CSS
st.markdown("""
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
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    try:
        st.session_state.agent, st.session_state.deps = create_msp_expert()
        st.session_state.agent_ready = True
    except Exception as e:
        st.session_state.agent_ready = False
        st.session_state.error = str(e)


# Sidebar
with st.sidebar:
    st.title("💾 MSP360 Backup Expert")
    st.markdown("---")
    
    st.markdown("""
    ### About
    AI assistant with access to the complete MSP360 Backup knowledge base documentation.
    
    ### Capabilities
    - Troubleshoot backup errors
    - Explain error codes
    - Provide step-by-step solutions
    - Configuration guidance
    - Best practices advice
    - Cloud storage setup help
    
    ### Documentation Sources
    - **help.msp360.com**: Official documentation
    - **kb.msp360.com**: Knowledge base articles
    - **Error codes**: Detailed error explanations
    - **Troubleshooting**: Step-by-step guides
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about MSP360 Backup..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Searching documentation...")
        
        try:
            # Run agent (simpler non-streaming version for stability)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                st.session_state.agent.run(
                    prompt,
                    deps=st.session_state.deps
                )
            )
            loop.close()
            
            response = result.data
            
            # Display response
            message_placeholder.markdown(response)
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    Powered by Pydantic AI + OpenAI (LLM) + Gemini (Embeddings) + Qdrant | 
    <a href='https://help.msp360.com' target='_blank'>MSP360 Documentation</a> | 
    <a href='https://kb.msp360.com' target='_blank'>Knowledge Base</a>
</div>
""", unsafe_allow_html=True)

