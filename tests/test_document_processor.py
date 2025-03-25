"""
Unit tests for the document_processor module.
"""

import unittest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

import document_processor


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for the document processor module."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample documents for testing
        self.sample_documents = [
            Document(page_content="This is test document 1", metadata={"source": "test1"}),
            Document(page_content="This is test document 2", metadata={"source": "test2"}),
        ]
        
        # Sample URL for testing
        self.test_url = "https://example.com"

    @patch('document_processor.WebBaseLoader')
    def test_load_documents_from_url(self, mock_web_loader):
        """Test loading documents from a URL."""
        # Configure the mock
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = self.sample_documents
        mock_web_loader.return_value = mock_loader_instance
        
        # Call the function
        result = document_processor.load_documents_from_url(self.test_url)
        
        # Assertions
        mock_web_loader.assert_called_once()
        self.assertEqual(result, self.sample_documents)
        self.assertEqual(len(result), 2)

    def test_split_documents(self):
        """Test splitting documents into chunks."""
        # Create a longer document for splitting
        long_document = Document(
            page_content="This is a longer document that should be split into multiple chunks. " * 10,
            metadata={"source": "long_doc"}
        )
        
        # Call the function
        result = document_processor.split_documents([long_document])
        
        # Assertions
        self.assertGreater(len(result), 1, "Document should be split into multiple chunks")
        for chunk in result:
            self.assertIsInstance(chunk, Document)
            self.assertLessEqual(len(chunk.page_content), 1000 + 200)  # chunk_size + overlap

    @patch('document_processor.HuggingFaceEmbeddings')
    @patch('document_processor.FAISS')
    def test_create_vector_store(self, mock_faiss, mock_embeddings):
        """Test creating a vector store from documents."""
        # Configure the mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vector_store = MagicMock(spec=VectorStore)
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # Call the function
        result = document_processor.create_vector_store(self.sample_documents)
        
        # Assertions
        mock_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        mock_faiss.from_documents.assert_called_once_with(self.sample_documents, mock_embeddings_instance)
        self.assertEqual(result, mock_vector_store)

    @patch('document_processor.HuggingFaceEmbeddings')
    @patch('document_processor.FAISS')
    def test_create_vector_store_with_fallback(self, mock_faiss, mock_embeddings):
        """Test creating a vector store with fallback to default embeddings."""
        # Configure the mocks to raise an exception on first call
        mock_embeddings.side_effect = [Exception("Test error"), MagicMock()]
        
        mock_vector_store = MagicMock(spec=VectorStore)
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # Call the function
        result = document_processor.create_vector_store(self.sample_documents)
        
        # Assertions
        self.assertEqual(mock_embeddings.call_count, 2)
        mock_faiss.from_documents.assert_called_once()
        self.assertEqual(result, mock_vector_store)

    @patch('document_processor.load_documents_from_url')
    @patch('document_processor.split_documents')
    @patch('document_processor.create_vector_store')
    def test_load_and_process_url(self, mock_create_vector_store, mock_split_documents, mock_load_documents):
        """Test the full document processing pipeline."""
        # Configure the mocks
        mock_load_documents.return_value = self.sample_documents
        mock_split_documents.return_value = self.sample_documents  # For simplicity
        mock_vector_store = MagicMock(spec=VectorStore)
        mock_create_vector_store.return_value = mock_vector_store
        
        # Call the function
        result = document_processor.load_and_process_url(self.test_url)
        
        # Assertions
        mock_load_documents.assert_called_once_with(self.test_url)
        mock_split_documents.assert_called_once_with(self.sample_documents)
        mock_create_vector_store.assert_called_once_with(self.sample_documents)
        self.assertEqual(result, mock_vector_store)


if __name__ == '__main__':
    unittest.main()
