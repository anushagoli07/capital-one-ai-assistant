import pytest
from core.document_processor import DocumentProcessor
import os

def test_chunking_logic():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    text = "This is a long sentence that should be split into multiple chunks for better RAG performance. This is the second sentence."
    
    # Create a dummy text file to simulate extraction if needed, 
    # but here we test the splitter directly
    chunks = processor.text_splitter.split_text(text)
    
    assert len(chunks) > 1
    assert "better RAG performance" in chunks[0] or "better RAG performance" in chunks[1]

def test_document_metadata():
    processor = DocumentProcessor()
    text = "Sample text for chunking."
    metadata = {"source": "test.pdf"}
    
    # Mocking the internal method to avoid file dependency
    processor.extract_text_from_pdf = lambda x: text
    
    docs = processor.process_document("dummy.pdf", metadata=metadata)
    
    assert len(docs) > 0
    assert docs[0].metadata["source"] == "test.pdf"
    assert "chunk_index" in docs[0].metadata
