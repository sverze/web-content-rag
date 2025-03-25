"""
RAG engine module for the application.
Handles the creation and querying of the RAG chain.
"""

from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_core.vectorstores import VectorStore
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    """State for the RAG application."""
    question: str
    context: List[Document]
    answer: str

def create_rag_chain(vector_store: VectorStore) -> Any:
    """
    Create a RAG chain using LangGraph.
    
    Args:
        vector_store: The vector store containing the indexed documents
        
    Returns:
        A compiled LangGraph for RAG
    """
    # Define the retrieve function
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    # Define the generate function
    def generate(state: State):
        # Get a prompt from the hub
        prompt = hub.pull("rlm/rag-prompt")
        
        # Join the document contents
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Create messages for the model
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        
        # Initialize the Anthropic model (Claude 3 Opus)
        llm = ChatAnthropic(temperature=0.5, max_tokens=1000, model_name="claude-3-opus-20240229")
        
        # Generate the response
        response = llm.invoke(messages)
        
        return {"answer": response.content}
    
    # Create and compile the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    # Define the edges
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    # Compile the graph
    return graph_builder.compile()

def query_rag_chain(rag_chain: Any, question: str) -> str:
    """
    Query the RAG chain with a question.
    
    Args:
        rag_chain: The RAG chain to query
        question: The question to ask
        
    Returns:
        The answer from the RAG chain
    """
    # Invoke the chain with the question
    result = rag_chain.invoke({"question": question})
    
    # Return the answer
    return result["answer"]
