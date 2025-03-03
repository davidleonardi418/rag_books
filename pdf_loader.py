import os
from typing import List, Dict
from pypdf import PdfReader
from tqdm import tqdm

class PDFLoader:
    def __init__(self, base_folder: str):
        """
        Initialize the PDF loader with the base folder containing textbooks.
        
        Args:
            base_folder: Path to the folder containing textbook PDFs
        """
        self.base_folder = base_folder
        
    def get_all_pdf_paths(self) -> List[str]:
        """
        Recursively find all PDF files in the base folder and subfolders.
        
        Returns:
            List of paths to PDF files
        """
        pdf_paths = []
        
        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
                    
        return pdf_paths
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Some PDFs might have issues with specific pages
            # Let's try to extract text page by page and skip problematic ones
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as page_error:
                    print(f"Warning: Could not extract text from page {i+1} in {pdf_path}: {page_error}")
                    # Continue with next page
                    continue
                
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            # For specific known errors, provide additional information
            if "Odd-length string" in str(e):
                print("  This error commonly occurs with malformed or corrupted PDF files.")
            elif "cryptography" in str(e):
                print("  This PDF may be encrypted. Ensure cryptography>=3.1 is installed.")
            return ""
    
    def load_textbooks(self) -> List[Dict]:
        """
        Load all textbooks from the base folder.
        
        Returns:
            List of dictionaries containing textbook metadata and content
        """
        pdf_paths = self.get_all_pdf_paths()
        textbooks = []
        
        print(f"Found {len(pdf_paths)} PDF files. Extracting text...")
        
        for pdf_path in tqdm(pdf_paths):
            relative_path = os.path.relpath(pdf_path, self.base_folder)
            category = os.path.dirname(relative_path)
            filename = os.path.basename(pdf_path)
            
            text_content = self.extract_text_from_pdf(pdf_path)
            
            if text_content:
                textbooks.append({
                    "path": pdf_path,
                    "category": category,
                    "filename": filename,
                    "content": text_content
                })
        
        print(f"Successfully loaded {len(textbooks)} textbooks")
        return textbooks 