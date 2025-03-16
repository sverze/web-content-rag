# Web Content Q&A RAG Application

A Retrieval Augmented Generation (RAG) application that allows users to ask questions about the content of any web page.

## Overview

This application uses LangChain, Claude AI, and Streamlit to create a powerful question-answering system for web content. It follows the RAG pattern:

1. **Retrieval**: Loads content from a URL, processes it, and stores it in a vector database
2. **Augmentation**: Retrieves relevant content based on user questions
3. **Generation**: Uses Claude AI to generate accurate answers based on the retrieved content

## Features

- Load and process content from any web URL
- Split content into manageable chunks for better processing
- Create embeddings using HuggingFace models
- Store and index content in a FAISS vector database
- Ask questions about the content in natural language
- Get AI-generated answers based on the actual content of the page
- User-friendly web interface built with Streamlit

## Components

- **document_processor.py**: Handles loading, splitting, and indexing documents from URLs
- **rag_engine.py**: Creates and manages the RAG chain using LangGraph
- **main.py**: Command-line interface for the application
- **app.py**: Streamlit web interface for the application

## Requirements

- Python 3.8+
- LangChain and related packages
- Anthropic API key (for Claude AI)
- Streamlit (for web interface)
- FAISS (for vector storage)
- HuggingFace sentence-transformers

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the application from the command line:

```bash
python main.py
```

Or specify a URL directly:

```bash
python main.py --url https://example.com
```

You'll be prompted to enter your Anthropic API key if it's not set in your environment variables.

### Web Interface

Run the Streamlit web interface:

```bash
streamlit run app.py YOUR_ANTHROPIC_API_KEY
```

Or set the API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=your_api_key
streamlit run app.py
```

The web interface will open in your browser, where you can:
1. Enter a URL to analyze
2. Wait for the content to be processed
3. Ask questions about the content

## How It Works

1. **Document Loading**: The application fetches the content of the specified URL using WebBaseLoader
2. **Document Splitting**: The content is split into smaller chunks using RecursiveCharacterTextSplitter
3. **Vector Store Creation**: The chunks are embedded using HuggingFace embeddings and stored in a FAISS vector store
4. **RAG Chain**: A LangGraph is created to handle the retrieval and generation steps
5. **Question Answering**: When a question is asked, the application:
   - Retrieves relevant document chunks from the vector store
   - Passes the question and retrieved context to Claude AI
   - Returns the generated answer

## License

[MIT License](LICENSE)
