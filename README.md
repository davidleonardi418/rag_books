# RAG Books

A Retrieval-Augmented Generation (RAG) application for analyzing textbooks stored as PDFs using Python and Ollama.

## Project Overview

This application allows you to:
- Load and process PDF textbooks
- Create searchable indices from textbook content
- Query your textbook collection using natural language
- Get AI-generated responses based on the content of your textbooks

## Features

- **Modular Design**: Clear separation of concerns with well-defined components
- **Complete Pipeline**: Handles the entire RAG workflow from document loading to response generation
- **Metadata Preservation**: Maintains document source information throughout the pipeline
- **Flexible Interface**: Multiple modes of operation (build index, query, interactive)
- **Persistence**: Ability to save and load vector stores for future use
- **Integrated Ollama**: Runs Ollama inside the Docker container for seamless operation
- **Network Resilience**: Handles timeouts and retries for better reliability with fallback mechanisms
- **Memory Efficiency**: Suitable for systems with limited memory (e.g., 8GB MacBook Air)

## Docker Setup

This project runs in a Docker container to ensure consistent environment and avoid memory issues. The Docker container includes Ollama, so you don't need to install it separately.

### Prerequisites

- Docker installed on your system
- Textbooks in PDF format

## Quick Start

```bash
# Make the script executable (first time only)
chmod +x docker-run.sh

# Build the Docker image
./docker-run.sh build

# Run the container to build an index
./docker-run.sh run build --textbooks /textbooks --output /app/output/index.pkl

# Run the container to query the index with default model (llama3)
./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"

# Run the container with a specific model
OLLAMA_MODEL=mistral ./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"

# Run the container in interactive mode
./docker-run.sh run interactive --index /app/output/index.pkl
```

## Advanced Usage

### Customizing Environment Variables

```bash
# Build the Docker image
HTTP_TIMEOUT=1800 DISABLE_HF_TRANSFER=true ./docker-run.sh clean-rebuild

# Customize textbooks path
INDEX_PATH="$HOME/.rag-index" MODELS_PATH="$HOME/.rag-models" OLLAMA_MODEL="tinyllama" TEXTBOOKS_PATH="$HOME/Google Drive/My Drive/Books/Test" ./docker-run.sh run build --textbooks /textbooks --output /index/index.pkl

# Customize Ollama model
INDEX_PATH="$HOME/.rag-index" MODELS_PATH="$HOME/.rag-models" OLLAMA_MODEL="tinyllama" ./docker-run.sh run query --index /app/output/index.pkl --model tinyllama --question "What is deep learning?"

# Customize HTTP timeout for slow connections
HTTP_TIMEOUT="1200" ./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"

# Disable HF Transfer for more reliable downloads
DISABLE_HF_TRANSFER=true ./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"

# Prevent model redownloads (use existing models only)
NO_MODEL_DOWNLOAD=true ./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"

# Combine multiple settings
TEXTBOOKS_PATH="/path/to/books" OLLAMA_MODEL="mistral" HTTP_TIMEOUT="1200" DISABLE_HF_TRANSFER=true \
./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"
```

### External Storage for Models and Indices

By default, models and index files are stored within the Docker container and are lost when the container is removed. To persist these files:

#### Storing LLM Models Externally

```bash
# Store models in an external directory
MODELS_PATH="$HOME/.rag-models" ./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"

# Store models externally and prevent redownloads
MODELS_PATH="$HOME/.rag-models" NO_MODEL_DOWNLOAD=true ./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"
```

#### Storing Indices Externally

```bash
# Build and store index in an external directory
INDEX_PATH="$HOME/.rag-index" ./docker-run.sh run build --textbooks /textbooks --output /index/index.pkl

# Query using the external index
INDEX_PATH="$HOME/.rag-index" ./docker-run.sh run query --index /index/index.pkl --question "What is deep learning?"
```

#### Using Both External Storage Options

```bash
# Use both external storage options
MODELS_PATH="$HOME/.rag-models" INDEX_PATH="$HOME/.rag-index" \
./docker-run.sh run query --index /index/index.pkl --question "What is deep learning?"
```

#### Benefits of External Storage

- **Persistence**: Models and indices remain available even after container removal
- **Faster Rebuilds**: No need to re-download models when rebuilding the container
- **Disk Space Efficiency**: Models are stored only once on your host system
- **Sharing**: Models and indices can be shared between different projects

### Mounting External Directories

#### For macOS:
```bash
# Mount Google Drive directory
TEXTBOOKS_PATH="$HOME/Google Drive/My Drive/Books" ./docker-run.sh run build --textbooks /textbooks --output /app/output/index.pkl
```

#### For Linux:
```bash
# Mount Google Drive directory (if using Google Drive for Desktop)
TEXTBOOKS_PATH="/path/to/mounted/google-drive/Books" ./docker-run.sh run build --textbooks /textbooks --output /app/output/index.pkl
```

## Building and Running with Docker

### Handling Network Timeouts

If you encounter timeout errors during the build or runtime, you can use the following options:

1. **Rebuild with longer timeouts and skip model download during build:**
   ```bash
   HTTP_TIMEOUT=1200 DISABLE_HF_TRANSFER=true BUILD_DOWNLOAD_MODELS=false ./docker-run.sh rebuild
   ```
   This will:
   - Set a longer timeout (1200 seconds instead of the default 600)
   - Disable the Hugging Face transfer tool
   - Skip downloading models during the build process (they'll be downloaded at runtime)

2. **Run with longer runtime timeouts:**
   ```bash
   HTTP_TIMEOUT=1200 DISABLE_HF_TRANSFER=true ./docker-run.sh run query --index /app/output/index.pkl --question "What is deep learning?"
   ```

If you're using external storage for models, the first download might take longer, but subsequent runs will use the cached model.

### Troubleshooting

- **Network issues**: If you encounter timeout errors, first try rebuilding with longer timeouts and disabling Hugging Face transfer as shown above.
- **Build fails with model download errors**: Use `BUILD_DOWNLOAD_MODELS=false` when rebuilding to skip the model download during build.
- **Runtime model download errors**: Make sure your container has internet access and try increasing the HTTP_TIMEOUT value.

## Project Structure

- `main.py`: Main entry point for the application
- `pdf_loader.py`: Extracts text from PDF files
- `document_processor.py`: Splits documents into manageable chunks
- `vector_store.py`: Creates embeddings and enables semantic search
- `ollama_interface.py`: Connects to Ollama for generating responses
- `simple_indexer.py`: Memory-efficient document indexing using TF-IDF
- `ollama_wrapper.py`: Integration with Ollama for generative AI
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker configuration with integrated Ollama
- `docker-run.sh`: Helper script for Docker operations

## Manual Docker Commands

If you prefer to run Docker commands directly:

```bash
# Build the image
docker build -t rag-books .

# Build with custom timeouts
docker build --build-arg HTTP_TIMEOUT=1200 --build-arg DISABLE_HF_TRANSFER=true --no-cache -t rag-books .

# Run with external storage volumes
docker run -it \
  -v "/path/to/textbooks:/textbooks" \
  -v "$(pwd):/app/output" \
  -v "$HOME/.rag-models:/models" \
  -v "$HOME/.rag-index:/index" \
  -e OLLAMA_MODEL="llama3" \
  -e HTTP_TIMEOUT="1200" \
  -e DISABLE_HF_TRANSFER="true" \
  -e MODELS_PATH="/models" \
  -e INDEX_PATH="/index" \
  --network host \
  rag-books main.py query --index /app/output/index.pkl --question "What is deep learning?"
```