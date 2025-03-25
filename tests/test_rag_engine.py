"""
Unit tests for the rag_engine module.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_engine import create_rag_chain, query_rag_chain, State


class TestRagEngine(unittest.TestCase):
    """Test cases for the RAG engine module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock vector store
        self.mock_vector_store = MagicMock(spec=VectorStore)
        
        # Sample documents for testing
        self.sample_docs = [
            Document(page_content="This is a test document about AI."),
            Document(page_content="RAG stands for Retrieval Augmented Generation.")
        ]
        
        # Configure the mock to return sample documents
        self.mock_vector_store.similarity_search.return_value = self.sample_docs

    @patch('rag_engine.ChatAnthropic')
    @patch('rag_engine.hub')
    def test_create_rag_chain(self, mock_hub, mock_chat_anthropic):
        """Test that a RAG chain can be created successfully."""
        # Configure mocks
        mock_prompt = MagicMock()
        mock_hub.pull.return_value = mock_prompt
        
        # Create the chain
        chain = create_rag_chain(self.mock_vector_store)
        
        # Assert that the chain was created
        self.assertIsNotNone(chain)
        mock_hub.pull.assert_called_once_with("rlm/rag-prompt")

    @patch('rag_engine.ChatAnthropic')
    @patch('rag_engine.hub')
    def test_query_rag_chain(self, mock_hub, mock_chat_anthropic):
        """Test querying the RAG chain."""
        # Configure mocks
        mock_prompt = MagicMock()
        mock_hub.pull.return_value = mock_prompt
        
        # Configure the LLM mock
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a test answer about AI and RAG."
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat_anthropic.return_value = mock_llm_instance
        
        # Create the chain
        chain = create_rag_chain(self.mock_vector_store)
        
        # Query the chain
        result = query_rag_chain(chain, "What is RAG?")
        
        # Assertions
        self.assertEqual(result, "This is a test answer about AI and RAG.")
        self.mock_vector_store.similarity_search.assert_called_once_with("What is RAG?")

    def test_state_typing(self):
        """Test that the State TypedDict works correctly."""
        # Create a valid state
        state = {
            "question": "What is RAG?",
            "context": self.sample_docs,
            "answer": "RAG is Retrieval Augmented Generation"
        }
        
        # This is just a type check, so we're just ensuring we can create a valid State
        self.assertIsInstance(state, dict)
        self.assertEqual(state["question"], "What is RAG?")
        self.assertEqual(len(state["context"]), 2)
        self.assertEqual(state["answer"], "RAG is Retrieval Augmented Generation")


if __name__ == '__main__':
    unittest.main()
