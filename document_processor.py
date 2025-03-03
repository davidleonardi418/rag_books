from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def process_textbooks(self, textbooks: List[Dict]) -> List[Dict]:
        """
        Process textbooks by splitting them into chunks.
        
        Args:
            textbooks: List of textbook dictionaries
            
        Returns:
            List of document chunks with metadata
        """
        document_chunks = []
        
        for textbook in textbooks:
            chunks = self.text_splitter.split_text(textbook["content"])
            
            for i, chunk in enumerate(chunks):
                document_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "path": textbook["path"],
                        "category": textbook["category"],
                        "filename": textbook["filename"],
                        "chunk_id": i
                    }
                })
        
        print(f"Created {len(document_chunks)} document chunks")
        return document_chunks 