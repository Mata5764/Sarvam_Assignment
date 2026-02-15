"""Streamlit UI for the Deep Research Agent."""
import streamlit as st
import logging
import asyncio

from agent import ResearchAgent
from storage import SessionManager
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Deep Research Agent", page_icon="🔍", layout="wide")


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = ResearchAgent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []


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


def create_new_session():
    """Create a new research session."""
    session_id = st.session_state.session_manager.create_session(
        metadata={"created_from": "streamlit"}
    )
    st.session_state.session_id = session_id
    st.session_state.messages = []
    st.session_state.research_history = []
    logger.info(f"Created new session: {session_id}")
    return session_id


def load_session(session_id: str):
    """Load an existing session."""
    st.session_state.session_id = session_id
    
    # Load conversation history
    messages = st.session_state.session_manager.get_conversation_history(session_id)
    st.session_state.messages = messages
    
    # Load research history
    research = st.session_state.session_manager.get_research_history(session_id, limit=10)
    st.session_state.research_history = research
    
    logger.info(f"Loaded session: {session_id}")


def render_sidebar():
    """Render sidebar with session management."""
    with st.sidebar:
        st.title("🔍 Deep Research Agent")
        st.markdown("---")
        
        st.write(f"**LLM:** {config.LLM_MODEL}")
        st.write(f"**Search:** {config.SEARCH_PROVIDER}")
        st.markdown("---")
        
        # Session management
        if st.session_state.session_id:
            st.success(f"Active: {st.session_state.session_id[:8]}...")
            if st.button("New Session"):
                create_new_session()
                st.rerun()
        else:
            if st.button("Start New Session", type="primary"):
                create_new_session()
                st.rerun()


def render_chat_interface():
    """Render main chat interface."""
    st.title("🔍 Deep Research Agent")
    
    if not st.session_state.session_id:
        st.info("👈 Please start a new session from the sidebar.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message.get('role')
        content = message.get('content')
        
        with st.chat_message(role):
            st.markdown(content)
    
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
        
        # Save to database
        st.session_state.session_manager.add_user_message(
            st.session_state.session_id,
            prompt
        )
        
        # Generate response
        with st.chat_message("assistant"):
            progress_placeholder = st.empty()
            answer_placeholder = st.empty()
            
            try:
                # Progress callback
                def show_progress(msg):
                    progress_placeholder.info(msg)
                
                # Get conversation context
                conversation_history = st.session_state.session_manager.get_context_for_query(
                    st.session_state.session_id,
                    max_messages=10
                )
                
                # Research (async)
                result = asyncio.run(
                    st.session_state.agent.research(
                        query=prompt,
                        conversation_history=conversation_history,
                        progress_callback=show_progress
                    )
                )
                
                # Clear progress
                progress_placeholder.empty()
                
                # Display answer
                answer = result.answer.answer
                answer_placeholder.markdown(answer)
                
                # Display citations
                if result.answer.citations:
                    st.markdown("**📚 Sources:**")
                    for i, citation in enumerate(result.answer.citations, 1):
                        st.markdown(f"{i}. [{citation.title}]({citation.url}) — {citation.domain}")
                    st.caption(f"Confidence: {result.answer.confidence}")
                
                # Conflict warning
                if result.answer.conflicts_detected:
                    st.warning(result.answer.conflict_note)
                
                # Save to database
                st.session_state.session_manager.add_assistant_message(
                    st.session_state.session_id, answer
                )
                
                # Save research turn
                st.session_state.session_manager.add_research_turn(
                    session_id=st.session_state.session_id,
                    query=prompt,
                    search_queries=[],
                    urls_opened=result.urls_used or [],
                    context_snippets=[],
                    answer=answer,
                    citations=result.answer.citations,
                    confidence=result.answer.confidence
                )
                
                # Update session state
                st.session_state.messages.append({'role': 'assistant', 'content': answer})
                
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                error_msg = f"Error: {str(e)}"
                answer_placeholder.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})


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
