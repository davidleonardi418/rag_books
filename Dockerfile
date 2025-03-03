FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Accept build arguments
ARG HTTP_TIMEOUT=600
ARG DISABLE_HF_TRANSFER=false
ARG BUILD_DOWNLOAD_MODELS=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install hf_transfer for faster downloads
RUN pip install --no-cache-dir hf_transfer

# Set environment variables for better download performance
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=${HTTP_TIMEOUT}
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV TOKENIZERS_PARALLELISM=true
ENV HF_HUB_USER_AGENT="rag-books/1.0"

# Disable HF Transfer if requested
RUN if [ "$DISABLE_HF_TRANSFER" = "true" ]; then \
    echo "Disabling HF Transfer for build process"; \
    export HF_HUB_ENABLE_HF_TRANSFER=0; \
    fi

# Pre-download sentence transformer model with extended timeout to avoid runtime downloads
RUN echo "Downloading model with timeout ${HTTP_TIMEOUT} seconds"

# Create Python script for model download with retry logic
RUN echo '#!/usr/bin/env python3' > /tmp/download_model.py && \
    echo 'from huggingface_hub import snapshot_download' >> /tmp/download_model.py && \
    echo 'import os' >> /tmp/download_model.py && \
    echo 'import pathlib' >> /tmp/download_model.py && \
    echo 'import sys' >> /tmp/download_model.py && \
    echo 'import time' >> /tmp/download_model.py && \
    echo '' >> /tmp/download_model.py && \
    echo '# Set timeout from argument' >> /tmp/download_model.py && \
    echo 'os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = sys.argv[1]' >> /tmp/download_model.py && \
    echo '' >> /tmp/download_model.py && \
    echo '# Define model directory' >> /tmp/download_model.py && \
    echo 'model_dir = "/app/.cache/huggingface/sentence-transformers/all-MiniLM-L6-v2"' >> /tmp/download_model.py && \
    echo '' >> /tmp/download_model.py && \
    echo '# Check if model already exists' >> /tmp/download_model.py && \
    echo 'if pathlib.Path(model_dir).exists():' >> /tmp/download_model.py && \
    echo '    print(f"Model already exists at {model_dir}, skipping download")' >> /tmp/download_model.py && \
    echo '    exit(0)' >> /tmp/download_model.py && \
    echo '' >> /tmp/download_model.py && \
    echo '# Try download with retry' >> /tmp/download_model.py && \
    echo 'max_retries = 3' >> /tmp/download_model.py && \
    echo 'for retry in range(max_retries):' >> /tmp/download_model.py && \
    echo '    try:' >> /tmp/download_model.py && \
    echo '        print(f"Downloading model to {model_dir} (Attempt {retry+1}/{max_retries})")' >> /tmp/download_model.py && \
    echo '        snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=model_dir)' >> /tmp/download_model.py && \
    echo '        print("Download successful")' >> /tmp/download_model.py && \
    echo '        exit(0)' >> /tmp/download_model.py && \
    echo '    except Exception as e:' >> /tmp/download_model.py && \
    echo '        print(f"Error during download: {e}")' >> /tmp/download_model.py && \
    echo '        if retry < max_retries - 1:' >> /tmp/download_model.py && \
    echo '            wait_time = 10 * (retry + 1)' >> /tmp/download_model.py && \
    echo '            print(f"Retrying in {wait_time} seconds...")' >> /tmp/download_model.py && \
    echo '            time.sleep(wait_time)' >> /tmp/download_model.py && \
    echo '        else:' >> /tmp/download_model.py && \
    echo '            print("Failed to download after multiple attempts")' >> /tmp/download_model.py && \
    echo '            if os.environ.get("BUILD_DOWNLOAD_MODELS", "true").lower() == "false":' >> /tmp/download_model.py && \
    echo '                print("BUILD_DOWNLOAD_MODELS is false, continuing without model")' >> /tmp/download_model.py && \
    echo '                exit(0)' >> /tmp/download_model.py && \
    echo '            else:' >> /tmp/download_model.py && \
    echo '                exit(1)' >> /tmp/download_model.py

# Execute download script conditionally
RUN chmod +x /tmp/download_model.py && \
    export BUILD_DOWNLOAD_MODELS=${BUILD_DOWNLOAD_MODELS} && \
    if [ "$BUILD_DOWNLOAD_MODELS" = "true" ]; then \
    echo "Attempting to download model during build" && \
    python /tmp/download_model.py "${HTTP_TIMEOUT}" || echo "WARNING: Model download failed, will download at runtime"; \
    else \
    echo "Skipping model download during build (BUILD_DOWNLOAD_MODELS=$BUILD_DOWNLOAD_MODELS)"; \
    fi && \
    rm /tmp/download_model.py

# Create initialization script to verify model downloads with proper escaping
RUN echo 'from sentence_transformers import SentenceTransformer' > /app/test_model.py && \
    echo 'import os' >> /app/test_model.py && \
    echo '' >> /app/test_model.py && \
    echo 'model_name = "all-MiniLM-L6-v2"' >> /app/test_model.py && \
    echo "print(f\"Loading model {model_name} with timeout {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', 'default')}\")" >> /app/test_model.py && \
    echo '' >> /app/test_model.py && \
    echo 'model = SentenceTransformer(model_name)' >> /app/test_model.py && \
    echo 'print(f"Successfully loaded model: {model}")' >> /app/test_model.py && \
    echo '' >> /app/test_model.py && \
    echo '# Test the model with a simple embedding' >> /app/test_model.py && \
    echo 'embedding = model.encode("This is a test sentence")' >> /app/test_model.py && \
    echo 'print(f"Model embedding shape: {embedding.shape}")' >> /app/test_model.py

RUN python /app/test_model.py

# Copy the rest of the application
COPY . .

# Create a volume for textbooks
VOLUME /textbooks

# Create volumes for external storage
VOLUME /models
VOLUME /index

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_MODELS_PATH=/models/ollama
ENV NO_MODEL_DOWNLOAD=false

# Create a script to start Ollama and then run the Python command
RUN echo '#!/bin/bash\n\
    # Disable HF_TRANSFER if it causes issues\n\
    if [ "$DISABLE_HF_TRANSFER" = "true" ]; then\n\
    export HF_HUB_ENABLE_HF_TRANSFER=0\n\
    echo "Disabled HF Transfer for downloads"\n\
    fi\n\
    \n\
    # Set HTTP timeout from environment variable\n\
    if [ -n "$HTTP_TIMEOUT" ]; then\n\
    export HF_HUB_DOWNLOAD_TIMEOUT=$HTTP_TIMEOUT\n\
    echo "Set HTTP timeout to $HTTP_TIMEOUT seconds"\n\
    fi\n\
    \n\
    # Set additional environment variables for network reliability\n\
    export TOKENIZERS_PARALLELISM=false\n\
    export TRANSFORMERS_OFFLINE=0\n\
    \n\
    # Configure model paths\n\
    if [ -n "$MODELS_PATH" ]; then\n\
    export OLLAMA_MODELS_PATH="$MODELS_PATH/ollama"\n\
    export SENTENCE_TRANSFORMERS_HOME="$MODELS_PATH/sentence_transformers"\n\
    export HUGGINGFACE_HUB_CACHE="$MODELS_PATH/huggingface"\n\
    export TRANSFORMERS_CACHE="$MODELS_PATH/huggingface"\n\
    mkdir -p "$OLLAMA_MODELS_PATH"\n\
    mkdir -p "$SENTENCE_TRANSFORMERS_HOME"\n\
    mkdir -p "$HUGGINGFACE_HUB_CACHE"\n\
    echo "Using models path: $MODELS_PATH"\n\
    fi\n\
    \n\
    # For debugging network issues\n\
    echo "Current network settings:"\n\
    echo " - HF_HUB_ENABLE_HF_TRANSFER: $HF_HUB_ENABLE_HF_TRANSFER"\n\
    echo " - HF_HUB_DOWNLOAD_TIMEOUT: $HF_HUB_DOWNLOAD_TIMEOUT"\n\
    echo " - TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"\n\
    echo " - OLLAMA_MODELS_PATH: $OLLAMA_MODELS_PATH"\n\
    echo " - SENTENCE_TRANSFORMERS_HOME: $SENTENCE_TRANSFORMERS_HOME"\n\
    echo " - HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"\n\
    echo " - NO_MODEL_DOWNLOAD: $NO_MODEL_DOWNLOAD"\n\
    \n\
    # Start Ollama in the background\n\
    ollama serve &\n\
    sleep 5\n\
    \n\
    # Check if Ollama is running\n\
    if ! curl -s http://localhost:11434/api/version >/dev/null; then\n\
    echo "Warning: Ollama is not responding. Waiting additional time..."\n\
    sleep 10\n\
    if ! curl -s http://localhost:11434/api/version >/dev/null; then\n\
    echo "Error: Ollama failed to start properly. Continuing anyway..."\n\
    else\n\
    echo "Ollama is now running."\n\
    fi\n\
    else\n\
    echo "Ollama is running."\n\
    fi\n\
    \n\
    # Check if model exists before pulling\n\
    if [ -n "$OLLAMA_MODEL" ] && [ "$NO_MODEL_DOWNLOAD" != "true" ]; then\n\
    if ollama list | grep -q "$OLLAMA_MODEL"; then\n\
    echo "Model $OLLAMA_MODEL already exists, skipping download"\n\
    else\n\
    echo "Pulling model: $OLLAMA_MODEL"\n\
    ollama pull $OLLAMA_MODEL\n\
    fi\n\
    elif [ -n "$OLLAMA_MODEL" ] && [ "$NO_MODEL_DOWNLOAD" = "true" ]; then\n\
    echo "Skipping model download as NO_MODEL_DOWNLOAD is set to true"\n\
    if ! ollama list | grep -q "$OLLAMA_MODEL"; then\n\
    echo "Warning: Model $OLLAMA_MODEL does not exist locally. Queries may fail."\n\
    fi\n\
    fi\n\
    \n\
    # Execute the Python command\n\
    exec python "$@"\n\
    ' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint to our script
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["main.py", "--help"] 