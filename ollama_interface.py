from typing import List, Dict
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class OllamaInterface:
    def __init__(self, model_name: str = "llama3"):
        """
        Initialize the Ollama interface.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.llm = OllamaLLM(model=model_name)
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an AI assistant analyzing textbooks. Use the following retrieved information to answer the question.
            
            Context information:
            {context}
            
            Question: {question}
            
            Provide a comprehensive answer based on the context information. If the information needed is not in the context, say so.
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def generate_response(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate a response to a question using retrieved documents.
        
        Args:
            question: User question
            retrieved_docs: List of retrieved documents
            
        Returns:
            Generated response
        """
        # Format context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} (from {doc['metadata']['filename']}, category: {doc['metadata']['category']}):\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Generate response
        response = self.chain.run(context=context, question=question)
        
        return response 