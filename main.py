#!/usr/bin/env python3
"""
Simple RAG (Retrieval Augmented Generation) application using LangChain.
This app allows users to provide a URL and ask questions about the content.
"""

import os
import argparse
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from document_processor import load_and_process_url, load_documents_from_url
from rag_engine import create_rag_chain, query_rag_chain

def main():
    """
    Main function to run the RAG application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG application for web content")
    parser.add_argument("--url", help="URL to process for RAG")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        api_key = input("Please enter your Anthropic API key: ")
        os.environ["ANTHROPIC_API_KEY"] = api_key

    # Initialize the RAG application
    if args.url:
        # Process the URL provided as argument
        vector_store = load_and_process_url(args.url)
        print("Document processing complete. Vector store created.")
        interactive_mode(vector_store)
    else:
        # Interactive mode from the beginning
        url = input("Please enter a URL to analyze: ")
        vector_store = load_and_process_url(url)
        print("Document processing complete. Vector store created.")
        interactive_mode(vector_store)

def interactive_mode(vector_store: VectorStore):
    """
    Run the interactive Q&A session with the user.
    
    Args:
        vector_store: The vector store containing the processed documents
    """
    # Create the RAG chain
    print("Creating RAG chain with Claude 3.7 Sonnet...")
    rag_chain = create_rag_chain(vector_store)
    
    print("\nRAG Application ready! Ask questions about the content or type 'exit' to quit.")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        print("Retrieving relevant information...")
        
        try:
            # Process the question through the RAG chain
            answer = query_rag_chain(rag_chain, question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"\nError processing your question: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
