#!/bin/bash

# Exit on error
set -e

# Build the Docker image
build_image() {
  echo "Building Docker image..."
  docker build -t rag-books .
}

# Build the Docker image with build arguments
build_image_with_args() {
  echo "Building Docker image with custom arguments..."
  echo "HTTP_TIMEOUT: $HTTP_TIMEOUT"
  echo "DISABLE_HF_TRANSFER: $DISABLE_HF_TRANSFER"
  
  docker build \
    --build-arg HTTP_TIMEOUT="$HTTP_TIMEOUT" \
    --build-arg DISABLE_HF_TRANSFER="$DISABLE_HF_TRANSFER" \
    --no-cache \
    -t rag-books .
}

# Clean and rebuild the Docker image (no cache)
clean_rebuild() {
  echo "Performing clean rebuild of Docker image..."
  echo "HTTP_TIMEOUT: $HTTP_TIMEOUT"
  echo "DISABLE_HF_TRANSFER: $DISABLE_HF_TRANSFER"
  
  # Remove existing image if it exists
  if docker image inspect rag-books >/dev/null 2>&1; then
    echo "Removing existing rag-books image..."
    docker rmi rag-books
  fi
  
  # Build with no cache
  docker build \
    --build-arg HTTP_TIMEOUT="$HTTP_TIMEOUT" \
    --build-arg DISABLE_HF_TRANSFER="$DISABLE_HF_TRANSFER" \
    --no-cache \
    -t rag-books .
}

# Run the container with the specified command
run_container() {
  local cmd="$1"
  shift
  
  echo "Running container with command: $cmd $@"
  echo "Environment settings:"
  echo "  - OLLAMA_MODEL: $OLLAMA_MODEL"
  echo "  - HTTP_TIMEOUT: $HTTP_TIMEOUT"
  echo "  - DISABLE_HF_TRANSFER: $DISABLE_HF_TRANSFER"
  echo "  - MODELS_PATH: $MODELS_PATH"
  echo "  - INDEX_PATH: $INDEX_PATH"
  echo "  - NO_MODEL_DOWNLOAD: $NO_MODEL_DOWNLOAD"
  
  # Create external directories if they don't exist
  if [ -n "$MODELS_PATH" ]; then
    mkdir -p "$MODELS_PATH"
    echo "Created models directory: $MODELS_PATH"
  fi
  
  if [ -n "$INDEX_PATH" ]; then
    mkdir -p "$INDEX_PATH"
    echo "Created index directory: $INDEX_PATH"
  fi
  
  # Prepare volume mounts
  local volume_mounts="-v \"$TEXTBOOKS_PATH:/textbooks\" -v \"$(pwd):/app/output\""
  
  # Add external model path if specified
  if [ -n "$MODELS_PATH" ]; then
    volume_mounts="$volume_mounts -v \"$MODELS_PATH:/models\""
  fi
  
  # Add external index path if specified
  if [ -n "$INDEX_PATH" ]; then
    volume_mounts="$volume_mounts -v \"$INDEX_PATH:/index\""
  fi
  
  # Build arguments array to properly preserve quotes
  args=()
  for arg in "$@"; do
    args+=("$arg")
  done
  
  # Build and execute docker run command
  local docker_cmd="docker run -it $volume_mounts \
    -e OLLAMA_MODEL=\"$OLLAMA_MODEL\" \
    -e HTTP_TIMEOUT=\"$HTTP_TIMEOUT\" \
    -e DISABLE_HF_TRANSFER=\"$DISABLE_HF_TRANSFER\" \
    -e MODELS_PATH=\"/models\" \
    -e INDEX_PATH=\"/index\" \
    -e NO_MODEL_DOWNLOAD=\"$NO_MODEL_DOWNLOAD\" \
    --network host \
    rag-books main.py \"$cmd\""
  
  # Add each argument with proper quoting
  for arg in "${args[@]}"; do
    docker_cmd="$docker_cmd \"$arg\""
  done
  
  echo "Executing: $docker_cmd"
  eval "$docker_cmd"
}

# Display usage information
usage() {
  echo "Usage: $0 [build|rebuild|clean-rebuild|run] [command] [args...]"
  echo ""
  echo "Commands:"
  echo "  build                Build the Docker image"
  echo "  rebuild              Rebuild the Docker image with custom timeouts (no cache)"
  echo "  clean-rebuild        Remove existing image and rebuild from scratch"
  echo "  run build            Run the container with 'build' command"
  echo "  run query            Run the container with 'query' command"
  echo "  run interactive      Run the container in interactive mode"
  echo ""
  echo "Environment variables:"
  echo "  TEXTBOOKS_PATH       Path to textbooks directory (default: $HOME/Google Drive/My Drive/Books/Machine Learning - Curated)"
  echo "  OLLAMA_MODEL         Ollama model to use (default: llama3)"
  echo "  HTTP_TIMEOUT         HTTP timeout in seconds for downloads (default: 600)"
  echo "  DISABLE_HF_TRANSFER  Set to 'true' to disable fast HF transfers if they cause issues (default: false)"
  echo "  MODELS_PATH          Path to models directory (optional)"
  echo "  INDEX_PATH           Path to index directory (optional)"
  echo "  NO_MODEL_DOWNLOAD    Set to 'true' to skip model downloads if models already exist (default: false)"
  echo ""
  echo "Examples:"
  echo "  $0 build"
  echo "  HTTP_TIMEOUT=1200 DISABLE_HF_TRANSFER=true $0 rebuild"
  echo "  HTTP_TIMEOUT=1800 DISABLE_HF_TRANSFER=true $0 clean-rebuild"
  echo "  $0 run build --textbooks /textbooks --output /app/output/index.pkl"
  echo "  $0 run query --index /app/output/index.pkl --question \"What is deep learning?\""
  echo "  OLLAMA_MODEL=llama3 $0 run query --index /app/output/index.pkl --question \"What is deep learning?\""
  echo "  DISABLE_HF_TRANSFER=true $0 run query --index /app/output/index.pkl --question \"What is machine learning?\""
  echo "  NO_MODEL_DOWNLOAD=true $0 run query --index /app/output/index.pkl --question \"What is deep learning?\""
  echo ""
  echo "External Storage Examples:"
  echo "  # Store models outside the container"
  echo "  MODELS_PATH=\"$HOME/.rag-models\" $0 run query --index /app/output/index.pkl --question \"What is deep learning?\""
  echo ""
  echo "  # Store index outside the container"
  echo "  INDEX_PATH=\"$HOME/.rag-index\" $0 run build --textbooks /textbooks --output /index/index.pkl"
  echo "  INDEX_PATH=\"$HOME/.rag-index\" $0 run query --index /index/index.pkl --question \"What is deep learning?\""
  echo ""
  echo "  # Use both external models and index directories"
  echo "  MODELS_PATH=\"$HOME/.rag-models\" INDEX_PATH=\"$HOME/.rag-index\" \\"
  echo "  $0 run query --index /index/index.pkl --question \"What is deep learning?\""
  exit 1
}

# Set default paths and models
TEXTBOOKS_PATH=${TEXTBOOKS_PATH:-"$HOME/Google Drive/My Drive/Books/Machine Learning - Curated"}
OLLAMA_MODEL=${OLLAMA_MODEL:-"llama3"}
HTTP_TIMEOUT=${HTTP_TIMEOUT:-"600"}
DISABLE_HF_TRANSFER=${DISABLE_HF_TRANSFER:-"false"}
MODELS_PATH=${MODELS_PATH:-""}
INDEX_PATH=${INDEX_PATH:-""}
NO_MODEL_DOWNLOAD=${NO_MODEL_DOWNLOAD:-"false"}
BUILD_DOWNLOAD_MODELS=${BUILD_DOWNLOAD_MODELS:-true}

# Main script logic
case "$1" in
  build)
    build_image
    ;;
  rebuild)
    echo "Rebuilding Docker image with HTTP_TIMEOUT=$HTTP_TIMEOUT, DISABLE_HF_TRANSFER=$DISABLE_HF_TRANSFER, BUILD_DOWNLOAD_MODELS=$BUILD_DOWNLOAD_MODELS"
    docker build \
        --build-arg HTTP_TIMEOUT="$HTTP_TIMEOUT" \
        --build-arg DISABLE_HF_TRANSFER="$DISABLE_HF_TRANSFER" \
        --build-arg BUILD_DOWNLOAD_MODELS="$BUILD_DOWNLOAD_MODELS" \
        -t rag-books .
    ;;
  clean-rebuild)
    clean_rebuild
    ;;
  run)
    if [ -z "$2" ]; then
      usage
    fi
    cmd="$2"
    shift 2
    run_container "$cmd" "$@"
    ;;
  *)
    usage
    ;;
esac 