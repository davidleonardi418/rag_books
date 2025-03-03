import os
import pickle
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import re
import logging
import gc
import time
import json
import requests
from requests.adapters import HTTPAdapter, Retry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure requests with longer timeouts and retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.request = lambda method, url, **kwargs: super(requests.Session, session).request(
    method=method, url=url, timeout=180, **kwargs
)

# Override the default session in Hugging Face Hub
try:
    import huggingface_hub.file_download
    huggingface_hub.file_download._get_session = lambda: session
except ImportError:
    logger.warning("Could not patch Hugging Face Hub session")

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        # Force CPU usage to avoid Metal/MPS issues on macOS
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Set longer timeout for Hugging Face downloads
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
        
        try:
            logger.info(f"Loading model {model_name} with timeout settings")
            # Create model with explicit device setting
            self.model = SentenceTransformer(model_name)
            # Force model to CPU
            self.model.to(torch.device('cpu'))
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Try loading with default model if possible (should be cached in Docker)
            logger.info("Falling back to cached model")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model.to(torch.device('cpu'))
        
        self.index = None
        self.documents = []
    
    def clean_text(self, text):
        """
        Clean text to make it suitable for embedding.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove null bytes and other control characters
        text = text.replace('\x00', ' ')
        text = re.sub(r'[\x01-\x1F\x7F]', ' ', text)
        
        # Replace excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Limit text length if extremely long
        if len(text) > 10000:  # Reduced from 50000 to 10000
            text = text[:10000]
            
        return text.strip()
        
    def add_documents(self, documents: List[Dict], output_path=None, checkpoint_dir="checkpoints"):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries
            output_path: Path to save the final index
            checkpoint_dir: Directory to save checkpoints
        """
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check if we have an existing checkpoint
        checkpoint_info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
        start_batch = 0
        
        if os.path.exists(checkpoint_info_path):
            try:
                with open(checkpoint_info_path, 'r') as f:
                    checkpoint_info = json.load(f)
                    start_batch = checkpoint_info.get('next_batch', 0)
                    
                # Load existing progress
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{start_batch-1}.pkl")
                if os.path.exists(checkpoint_path):
                    print(f"Loading checkpoint from batch {start_batch-1}...")
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                        self.documents = checkpoint_data.get('documents', [])
                        self.embeddings = checkpoint_data.get('embeddings', [])
                else:
                    print("Checkpoint file not found, starting from the beginning.")
                    start_batch = 0
                    self.documents = []
                    self.embeddings = []
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                start_batch = 0
                self.documents = []
                self.embeddings = []
        else:
            self.documents = []
            self.embeddings = []
        
        # Process in smaller batches
        batch_size = 8  # Reduced from 16 to 8
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"Processing {len(documents)} documents in {total_batches} batches (starting from batch {start_batch})...")
        
        for batch_idx in range(start_batch, total_batches):
            # Force garbage collection before processing each batch
            gc.collect()
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]
            
            print(f"Batch {batch_idx+1}/{total_batches}: Processing {len(batch_docs)} documents")
            
            # Clean and prepare texts
            batch_texts = []
            batch_indices = []
            
            for i, doc in enumerate(batch_docs):
                try:
                    cleaned_text = self.clean_text(doc["content"])
                    if cleaned_text:
                        batch_texts.append(cleaned_text)
                        batch_indices.append(i)
                except Exception as e:
                    print(f"Error cleaning text at position {start_idx + i}: {e}")
            
            if not batch_texts:
                print(f"Batch {batch_idx+1}/{total_batches}: No valid texts")
                
                # Save checkpoint info
                with open(checkpoint_info_path, 'w') as f:
                    json.dump({'next_batch': batch_idx + 1}, f)
                    
                continue
            
            # Create embeddings for this batch with even smaller sub-batches
            successful_docs = []
            successful_embeddings = []
            
            # Use sub-batches of 2 to minimize memory issues
            sub_batch_size = 2
            for sub_idx in range(0, len(batch_texts), sub_batch_size):
                sub_end = min(sub_idx + sub_batch_size, len(batch_texts))
                sub_texts = batch_texts[sub_idx:sub_end]
                sub_indices = batch_indices[sub_idx:sub_end]
                
                # Let's sleep a tiny bit between sub-batches to allow memory cleanup
                time.sleep(0.1)
                
                try:
                    # Process one text at a time for maximum safety
                    for j, (text, idx) in enumerate(zip(sub_texts, sub_indices)):
                        try:
                            # Get the original document
                            doc = batch_docs[idx]
                            
                            # Create embedding
                            embedding = self.model.encode([text], convert_to_numpy=True)[0]
                            
                            # Store successful result
                            successful_docs.append(doc)
                            successful_embeddings.append(embedding)
                            
                            print(f"  Successfully embedded text {sub_idx + j + 1}/{len(batch_texts)}")
                            
                        except Exception as e2:
                            print(f"  Error embedding text {sub_idx + j + 1}: {e2}")
                            
                except Exception as e:
                    print(f"  Error in sub-batch: {e}")
            
            # Add successful results to the main collections
            self.documents.extend(successful_docs)
            self.embeddings.extend(successful_embeddings)
            
            print(f"Batch {batch_idx+1}/{total_batches}: Added {len(successful_docs)} documents")
            
            # Save checkpoint after each batch
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings
                }, f)
            
            # Update checkpoint info
            with open(checkpoint_info_path, 'w') as f:
                json.dump({'next_batch': batch_idx + 1}, f)
            
            print(f"Saved checkpoint for batch {batch_idx+1}")
            
            # Force garbage collection
            gc.collect()
        
        # Create FAISS index from all collected embeddings
        if self.embeddings:
            print("Creating FAISS index from all embeddings...")
            embeddings_array = np.vstack(self.embeddings).astype('float32')
            vector_dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(vector_dimension)
            self.index.add(embeddings_array)
            
            print(f"Created index with {len(self.documents)} documents")
            
            # Clean up embeddings to free memory
            del self.embeddings
            gc.collect()
            
            # Save final index if path provided
            if output_path:
                self.save(output_path)
                
            # Clean up checkpoints
            print("Cleaning up checkpoint files...")
            for batch_idx in range(total_batches):
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}.pkl")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
            
            if os.path.exists(checkpoint_info_path):
                os.remove(checkpoint_info_path)
        else:
            print("No valid embeddings were created, index is empty")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document dictionaries
        """
        if not self.index:
            raise ValueError("Vector store is empty. Add documents first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search for similar documents
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(1.0 / (1.0 + distances[0][i]))
                results.append(doc)
        
        return results
    
    def save(self, path: str):
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the vector store
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'index': faiss.serialize_index(self.index) if self.index else None
            }, f)
        
        print(f"Vector store saved to {path}")
    
    def load(self, path):
        """
        Load a vector store from a file
        
        Args:
            path: Path to load the vector store from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        if 'index' in data and data['index'] is not None:
            self.index = faiss.deserialize_index(data['index'])
        else:
            self.index = None
            
        print(f"Vector store loaded from {path}") 