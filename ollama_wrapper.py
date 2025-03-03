"""
Wrapper to connect the simple search engine with Ollama for RAG.
"""
import os
import sys
import argparse
import pickle
import subprocess
import json
import time

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def generate_response(model_name, question, context):
    """Generate a response using Ollama."""
    prompt = f"""
You are an AI assistant analyzing textbooks. Use the following retrieved information to answer the question.

Context information:
{context}

Question: {question}

Provide a comprehensive answer based on the context information. If the information needed is not in the context, say so.
"""
    
    try:
        result = subprocess.run(
            ["ollama", "run", model_name], 
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            print(f"Error running Ollama: {result.stderr.decode()}")
            return "Error generating response with Ollama."
            
        return result.stdout.decode()
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

def query_with_ollama(index_path, question, model_name="llama3", num_results=5):
    """Query the search system and generate response with Ollama."""
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("Ollama is not installed or not in PATH. Please install Ollama first.")
        sys.exit(1)
    
    # Import and load search engine
    try:
        from simple_indexer import SimpleSearchEngine
    except ImportError:
        print("Could not import SimpleSearchEngine. Make sure simple_indexer.py is in the same directory.")
        sys.exit(1)
    
    # Load search engine
    search_engine = SimpleSearchEngine()
    search_engine.load(index_path)
    
    # Search for relevant documents
    retrieved_docs = search_engine.search(question, k=num_results)
    
    # Format context
    context = "\n\n".join([
        f"Document {i+1} (from {doc['metadata']['filename']}, category: {doc['metadata']['category']}):\n{doc['content']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    # Generate response
    response = generate_response(model_name, question, context)
    
    # Print results
    print("\n" + "="*50)
    print("Question:", question)
    print("="*50)
    print("Response:", response)
    print("="*50)
    
    # Print sources
    print("\nSources:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i+1}. {doc['metadata']['filename']} (Category: {doc['metadata']['category']})")
    
    return response

def interactive_mode(index_path, model_name="llama3", num_results=5):
    """Run in interactive mode."""
    print("\nTextbook Analysis RAG System - Interactive Mode")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            break
            
        query_with_ollama(index_path, question, model_name, num_results)

def main():
    parser = argparse.ArgumentParser(description="Ollama RAG Interface for Textbook Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
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
    
    if args.command == "query":
        query_with_ollama(args.index, args.question, args.model, args.num_results)
    elif args.command == "interactive":
        interactive_mode(args.index, args.model, args.num_results)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 