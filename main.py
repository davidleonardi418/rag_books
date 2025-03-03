import os
import argparse
from pdf_loader import PDFLoader
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ollama_interface import OllamaInterface

def build_index(textbooks_folder, output_path):
    """Build the vector index from textbooks."""
    # Load textbooks
    loader = PDFLoader(textbooks_folder)
    textbooks = loader.load_textbooks()
    
    # Process documents
    print("Processing documents...")
    processor = DocumentProcessor()
    document_chunks = processor.process_textbooks(textbooks)
    
    # Create vector store
    print("Creating vector store...")
    vector_store = VectorStore()
    # Use checkpoint-based document processing with output path
    vector_store.add_documents(document_chunks, output_path=output_path)
    
    print(f"Index built successfully and saved to {output_path}")

def query_system(index_path, question, model_name, num_results=5):
    """Query the RAG system with a question."""
    # Load vector store
    vector_store = VectorStore()
    vector_store.load(index_path)
    
    # Search for relevant documents
    retrieved_docs = vector_store.search(question, k=num_results)
    
    # Generate response using Ollama
    ollama = OllamaInterface(model_name=model_name)
    response = ollama.generate_response(question, retrieved_docs)
    
    # Print response
    print("\n" + "="*50)
    print("Question:", question)
    print("="*50)
    print("Response:", response)
    print("="*50)
    
    # Print sources
    print("\nSources:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i+1}. {doc['metadata']['filename']} (Category: {doc['metadata']['category']})")
    
def interactive_mode(index_path, model_name, num_results=5):
    """Run the RAG system in interactive mode."""
    print("\nTextbook Analysis RAG System - Interactive Mode")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            break
            
        query_system(index_path, question, model_name, num_results)

def main():
    parser = argparse.ArgumentParser(description="Textbook Analysis RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build index command
    build_parser = subparsers.add_parser("build", help="Build the vector index from textbooks")
    build_parser.add_argument("--textbooks", required=True, help="Path to the folder containing textbooks")
    build_parser.add_argument("--output", default="index.pkl", help="Path to save the index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--index", required=True, help="Path to the index file")
    query_parser.add_argument("--question", required=True, help="Question to ask")
    query_parser.add_argument("--model", default="llama3", help="Ollama model to use")
    query_parser.add_argument("--num-results", type=int, default=5, help="Number of results to retrieve")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    interactive_parser.add_argument("--index", required=True, help="Path to the index file")
    interactive_parser.add_argument("--model", default="llama3", help="Ollama model to use")
    interactive_parser.add_argument("--num-results", type=int, default=5, help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_index(args.textbooks, args.output)
    elif args.command == "query":
        query_system(args.index, args.question, args.model, args.num_results)
    elif args.command == "interactive":
        interactive_mode(args.index, args.model, args.num_results)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 