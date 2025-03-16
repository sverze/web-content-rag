"""
Document processing module for the RAG application.
Handles loading, splitting, and indexing documents from URLs.
"""

import bs4
from typing import List, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS

def load_and_process_url(url: str) -> VectorStore:
    """
    Load content from a URL, process it, and create a vector store.
    
    Args:
        url: The URL to load and process
        
    Returns:
        A vector store containing the processed documents
    """
    # Load the documents
    documents = load_documents_from_url(url)
    
    # Split the documents
    document_chunks = split_documents(documents)
    
    # Create and return the vector store
    return create_vector_store(document_chunks)

def load_documents_from_url(url: str) -> List[Document]:
    """
    Load documents from a URL.
    
    Args:
        url: The URL to load
        
    Returns:
        A list of Document objects
    """
    print(f"Loading content from: {url}")
    
    # Use WebBaseLoader to fetch the content
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                ["p", "h1", "h2", "h3", "h4", "h5", "article", "section", "div", "main"]
            )
        )
    )
    
    documents = loader.load()
    
    # Print the content for demonstration
    if documents:
        print("\n--- Page Content ---\n")
        for doc in documents:
            print(doc.page_content[:1000] + "...\n" if len(doc.page_content) > 1000 else doc.page_content)
        print("\n--- End of Content ---\n")
    else:
        print("No content was loaded from the URL.")
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better processing.
    
    Args:
        documents: List of documents to split
        
    Returns:
        List of split document chunks
    """
    print("Splitting documents into chunks...")
    
    # Create a text splitter with appropriate chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    
    # Split the documents
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
    
    return split_docs

def create_vector_store(documents: List[Document]) -> VectorStore:
    """
    Create a vector store from documents.
    
    Args:
        documents: List of documents to index
        
    Returns:
        A vector store containing the indexed documents
    """
    print("Creating vector store from document chunks...")
    
    # Initialize the OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create a FAISS vector store from the documents
    vector_store = FAISS.from_documents(documents, embeddings)
    
    print(f"Vector store created with {len(documents)} document chunks.")
    
    return vector_store
