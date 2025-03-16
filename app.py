"""
Streamlit UI for the RAG application.
This app allows users to provide a URL and ask questions about the content.
"""

import os
import streamlit as st
from typing import Optional

from document_processor import load_and_process_url
from rag_engine import create_rag_chain, query_rag_chain

# Set page configuration
st.set_page_config(
    page_title="Web Content Q&A",
    page_icon="🔍",
    layout="wide",
)

# Add title and description
st.title("Web Content Q&A")
st.markdown("""
This application allows you to ask questions about the content of any web page.
1. Enter a URL
2. Wait for the content to be processed
3. Ask questions about the content
""")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# API Key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Anthropic API Key:", type="password")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("API Key set!")

# URL input and processing
url_input = st.text_input("Enter a URL to analyze:", placeholder="https://example.com")

if url_input and st.button("Process URL"):
    if not st.session_state.api_key_set:
        st.error("Please enter your Anthropic API Key in the sidebar first.")
    else:
        with st.spinner("Loading and processing the web page..."):
            try:
                st.session_state.vector_store = load_and_process_url(url_input)
                st.session_state.rag_chain = create_rag_chain(st.session_state.vector_store)
                st.success("Web page processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing the URL: {str(e)}")

# Q&A section
st.header("Ask Questions")

if st.session_state.vector_store is not None and st.session_state.rag_chain is not None:
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
    
    # Question input
    question = st.chat_input("Ask a question about the web page content")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    answer = query_rag_chain(st.session_state.rag_chain, question)
                    st.write(answer)
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
else:
    st.info("Please process a URL first to ask questions about its content.")

# Footer
st.markdown("---")
st.markdown("Powered by LangChain, Claude, and Streamlit")
