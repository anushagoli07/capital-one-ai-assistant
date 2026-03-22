import PyPDF2
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts raw text from a PDF file.
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
        return text

    def process_document(self, pdf_path: str, metadata: dict = None) -> List[Document]:
        """
        Extracts text and splits it into chunks.
        """
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            return []
            
        chunks = self.text_splitter.split_text(raw_text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
            
        return documents
