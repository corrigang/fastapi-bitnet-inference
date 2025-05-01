# BitNet Inference Web UI 🧠

A modern web interface for running Microsoft's BitNet models efficiently on CPU. This project provides a user-friendly way to download, manage, and run inference with 1-bit quantized language models.

![BitNet Inference UI](/app/static/imgs/bit-inference5.png)

## 🌟 Features

- **Easy Model Management**
  - One-click downloads from Hugging Face
  - Direct model uploads (GGUF format)
  - Real-time download progress tracking
  - Popular models quick access

- **Efficient Inference**
  - CPU-optimized inference
  - Support for 1-bit quantized models
  - Conversation mode
  - Adjustable parameters (temperature, max tokens)

- **Modern UI/UX**
  - Clean, responsive interface
  - Dark/Light theme support
  - Real-time status updates
  - System logs viewer

- **Technical Features**
  - FastAPI backend
  - Async model downloads
  - Automatic fallback mechanisms
  - Progress monitoring system
![BitNet Inference UI](/app/static/imgs/bit-inference1.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CPU with AVX2 support (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mindscope-world/bitnet-inference.git
cd bitnet-inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The web interface will be available at `http://localhost:8000`

## 💻 Usage

### Downloading Models

1. Navigate to the "Download Model" tab
2. Enter a model name or HuggingFace path (e.g., `microsoft/BitNet-b1.58-2B-4T`)
3. Click "Download Model"
4. Monitor the download progress in real-time

### Running Inference

1. Ensure a model is loaded
2. Enter your prompt in the text area
3. Adjust generation parameters if needed:
   - Temperature (0.1 - 1.5)
   - Max Tokens (10 - 2048)
   - Conversation Mode (on/off)
4. Click "Generate"
![BitNet Inference UI](/app/static/imgs/bit-inference6.png)

### Model Compatibility

The application supports various BitNet models, including:
- BitNet-b1.58-2B-4T
- bitnet_b1_58-large
- bitnet_b1_58-3B

## 🛠️ Technical Details

### Architecture

```
bitnet-inference/
├── app/
│   ├── static/
│   │   ├── imgs/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   └── models/
├── app.py
├── setup_env.py
├── simple_model_server.py
└── requirements.txt
```

### Key Components

- **FastAPI Backend**: Handles model management and inference requests
- **Async Downloads**: Non-blocking model downloads with progress tracking
- **Fallback System**: Automatic switching between optimized and standard inference
- **Theme System**: Dynamic theme switching with system preference detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contributors

- [@mindscope-world](https://github.com/mindscope-world) - Project Lead & Main Developer

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Microsoft BitNet](https://github.com/microsoft/BitNet) - For the original BitNet implementation
- [FastAPI](https://fastapi.tiangolo.com/) - For the excellent web framework
- [Hugging Face](https://huggingface.co/) - For model hosting and transformers library

## 📞 Support

For support, please open an issue in the GitHub repository or contact [@mindscope-world](https://github.com/mindscope-world).

## 🔮 Future Plans

- [ ] Add batch processing support
- [ ] Implement model fine-tuning interface
- [ ] Add more visualization options
- [ ] Support for custom quantization
- [ ] API documentation interface
- [ ] Docker deployment support

---

Made with ❤️ by [@mindscope-world](https://github.com/mindscope-world)