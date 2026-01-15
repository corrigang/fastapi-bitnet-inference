# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BitNet Inference Web UI - A FastAPI-based web interface for running Microsoft's BitNet 1-bit quantized language models on CPU.

## Commands

### Run the Application

```bash
# Direct start (development with auto-reload)
python app.py

# Using the run script (sets up BitNet repo and validates requirements)
python run.py --host 0.0.0.0 --port 8000

# Create a dummy model for testing
python run.py --create-dummy --dummy-size 125M
```

The web interface runs at `http://localhost:8000`.

### Setup Environment for a Model

```bash
python setup_env.py --model-dir app/models/<model_name> --quant-type i2_s
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### System Requirements

- Python 3.9+
- CMake 3.22+
- Clang or MSVC compiler (Windows)
- CPU with AVX2 support (recommended)

## Architecture

### Core Components

- **app.py** - FastAPI backend with endpoints for model management and inference. Handles:
  - Model downloading from Hugging Face (`/download-model`)
  - Direct GGUF file uploads (`/upload-model`)
  - Text generation (`/generate`)
  - Download progress tracking (`/download-progress`)

- **simple_model_server.py** - Fallback inference server using HuggingFace Transformers. Used when:
  - GGUF model files aren't available
  - The native BitNet C++ inference fails
  - Supports 4-bit/8-bit quantization and pipeline-based loading

- **setup_env.py** - Environment setup script that:
  - Validates system requirements (Python, CMake, compiler)
  - Clones Microsoft's BitNet repository
  - Builds the native inference engine

- **run.py** - Application launcher that orchestrates setup and starts the server

### Inference Flow

1. User downloads model via UI or uploads GGUF file
2. `app.py` stores model in `app/models/<model_name>/`
3. On inference request:
   - If `.gguf` file exists: uses native BitNet `run_inference.py`
   - Otherwise: falls back to `simple_model_server.py` with Transformers

### Frontend

- `app/templates/index.html` - Main Jinja2 template
- `app/static/js/main.js` - UI logic for inference and model management
- `app/static/js/theme.js` - Dark/light theme switching

### Model Storage

Models are stored in `app/models/` with each model in its own subdirectory. Supports:
- GGUF format (native BitNet inference)
- SafeTensors/PyTorch formats (Transformers fallback)
