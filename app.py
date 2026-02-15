"""Streamlit UI for the Deep Research Agent."""
import streamlit as st
import logging
import asyncio

from agent.orchestrator import ResearchAgent
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Deep Research Agent", page_icon="🔍", layout="wide")


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = ResearchAgent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = "streamlit_session"


def validate_config():
    """Validate configuration and display errors if needed."""
    missing = config.validate()
    if missing:
        st.error("⚠️ Missing required configuration:")
        for item in missing:
            st.error(f"  - {item}")
        st.info("""
        Please set the required environment variables:
        1. Create a `.env` file in the project root
        2. Add your API keys (see README for details)
        3. Restart the application
        """)
        st.stop()


def render_sidebar():
    """Render sidebar with info."""
    with st.sidebar:
        st.title("🔍 Deep Research Agent")
        st.markdown("---")
        
        st.write(f"**Search:** {config.SEARCH_PROVIDER}")
        st.markdown("---")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def render_chat_interface():
    """Render main chat interface."""
    st.title("🔍 Deep Research Agent")
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message.get('role')
        content = message.get('content')
        citations = message.get('citations', [])
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Display citations if present
            if role == 'assistant' and citations:
                with st.expander("📚 Sources", expanded=False):
                    for i, citation in enumerate(citations, 1):
                        st.markdown(f"{i}. [{citation['title']}]({citation['url']}) — {citation['domain']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a research question..."):
        # Add user message to UI
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to session state
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })
        
        # Generate response
        with st.chat_message("assistant"):
            # Progress tracking
            progress_container = st.empty()
            status_updates = []
            
            def update_progress(update_text: str):
                """Update progress display."""
                status_updates.append(update_text)
                # Display last 5 updates
                progress_container.markdown("\n\n".join([f"• {u}" for u in status_updates[-5:]]))
            
            try:
                # Get conversation history (last 5 turns)
                conversation_history = []
                for msg in st.session_state.messages[-10:]:  # Last 10 messages = 5 turns
                    if msg['role'] in ['user', 'assistant']:
                        conversation_history.append({
                            'role': msg['role'],
                            'content': msg['content']
                        })
                
                # Create progress callback
                def progress_callback(message: str):
                    update_progress(message)
                
                # Research (async)
                with st.status("🔍 Researching...", expanded=True) as status:
                    result = asyncio.run(
                        st.session_state.agent.research(
                            query=prompt,
                            conversation_history=conversation_history[:-1],  # Exclude current message
                            session_id=st.session_state.session_id,
                            progress_callback=progress_callback
                        )
                    )
                    status.update(label="✅ Research Complete!", state="complete")
                
                # Clear progress display
                progress_container.empty()
                
                # Display answer
                answer = result.answer.answer
                st.markdown(answer)
                
                # Display citations
                citations_data = []
                if result.answer.citations:
                    with st.expander("📚 Sources", expanded=True):
                        for i, citation in enumerate(result.answer.citations, 1):
                            st.markdown(f"{i}. [{citation.title}]({citation.url}) — {citation.domain}")
                            citations_data.append({
                                'title': citation.title,
                                'url': citation.url,
                                'domain': citation.domain
                            })
                
                # Update session state
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'citations': citations_data
                })
                
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': error_msg,
                    'citations': []
                })


def main():
    """Main application entry point."""
    # Initialize
    initialize_session_state()
    
    # Validate configuration
    validate_config()
    
    # Render UI
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
