# Gemma 3 & Whisper API with FastAPI and LangChain

This project provides a FastAPI server that uses LangChain with ChatOllama to serve Google's Gemma 3 model. The implementation is optimized for concurrent processing of multiple requests.

## Features

- REST API for interacting with Gemma 3 model
- Concurrent request handling using a thread pool
- LLM instance caching for improved performance
- Request tracking and status monitoring
- Configurable parameters (temperature, max tokens)
- Support for system prompts and conversation history
- Easy setup with Makefile commands

## Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com) (for local model serving)
- [Make](https://www.gnu.org/software/make/) (optional, for using Makefile commands)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gemma3-api.git
cd gemma3-api
```

### 2. Pull the Gemma 3 model using Ollama

First, ensure Ollama is installed on your system. Then run:

```bash
ollama run gemma3:1b
```

This will download and start the Gemma 3 1B model. Once it's running, you can exit the Ollama process (Ctrl+C) as it will continue running in the background.

### 3. Set up Python environment

You can set up the environment in two ways:

#### Using Make (Recommended)

```bash
# Create virtual environment and install dependencies
make setup
```

#### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Start the FastAPI server

#### Using Make

```bash
# First activate the virtual environment if not already active
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the server
make run
```

#### Manual Start

```bash
# Make sure the virtual environment is activated
python main.py
```

The API will be available at [http://localhost:8000](http://localhost:8000).

## Using the Makefile

The project includes a Makefile with several useful commands:

```bash
# Display available commands
make help

# Set up the Python environment
make setup

# Run the server
make run

# Run the server in debug mode (with auto-reload)
make debug

# Check if Ollama is running and Gemma model is available
make check-ollama

# Clean up Python cache files
make clean

# Run tests (when added)
make test
```

## API Endpoints

### Chat Completion

Send a POST request to `/v1/chat/completions` with your prompt:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me about quantum computing"}],
    "system_prompt": "You are a helpful AI assistant",
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### Request Status

Check the status of a specific request:

```bash
curl "http://localhost:8000/v1/requests/{request_id}"
```

### Available Models

Get information about available models:

```bash
curl "http://localhost:8000/v1/models"
```

### Health Check

Check if the API is running and get the number of active requests:

```bash
curl "http://localhost:8000/health"
```

## Performance Tuning

The server is configured to handle multiple concurrent requests efficiently:

- Thread pool size is set to `CPU_COUNT * 2` by default
- LLM instances are cached for repeated requests with the same parameters
- Request tracking allows monitoring of system load

You can adjust the `MAX_WORKERS` value in the code to optimize for your specific hardware.

## Troubleshooting

### Ollama Connection Issues

If the FastAPI server can't connect to Ollama, ensure:

1. Ollama is running in the background
2. You've pulled the Gemma 3 model (`ollama run gemma3:1b`)
3. The default Ollama host (`http://localhost:11434`) is accessible

Use the included check command:

```bash
make check-ollama
```

Or manually verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

This should return a list of available models.

## License

[MIT License](LICENSE)