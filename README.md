# Yt2txt

## Description

This project allows you to download YouTube videos, extract audio, and transcribe the audio files into text using the Faster Whisper library.

## Installation

### 1. Prerequisites

- **Python 3.8 or newer** must be installed. [Download Python](https://www.python.org/downloads/).
- A GPU compatible with CUDA (e.g., NVIDIA RTX 4090) is recommended for better performance on Windows/Linux.
- macOS users with Apple Silicon (M1/M2/M4) can use the CPU version optimized for Metal Performance Shaders (MPS).

### 2. Set Up a Virtual Environment

Create and activate a virtual environment:

#### On Windows
```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

#### On macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required libraries for your project:

## 📥 Installation
```bash
pip install -r requirements.txt
```

And install ffprobe + ffmpeg

### 4. Install PyTorch

#### On Windows/Linux with NVIDIA GPU (CUDA)
Install PyTorch with CUDA support. Replace `cu118` with your specific CUDA version if necessary.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### On macOS with Apple Silicon (MPS)
Install the MPS-optimized version of PyTorch:

```bash
pip install torch torchvision torchaudio
```

#### For CPU-only
For systems without a GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Verify Installation

Run the following script to ensure PyTorch is configured correctly:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available()); print(torch.__version__)"
```

- On Windows/Linux, `torch.cuda.is_available()` should return `True`.
- On macOS with MPS, `torch.backends.mps.is_available()` should return `True`.

---

## Usage

## 🚀 Usage
```python
from yt2txt import YoutubeTranscript

# Example of extraction
transcript = YoutubeTranscript('https://youtu.be/VIDEO_ID').get_transcript()
```

Run the project with the following command:

```bash
python your_script.py --url "https://youtube.com/..." -o output -m large-v3
```

- `--url`: The URL of the YouTube video or playlist.
- `-o`: The output directory for the transcribed files.
- `-m`: Whisper model size to use (e.g., `large-v3`).

---

## Features

## 🌟 Features
- Subtitle extraction
- Export in TXT/JSON format
- Automatic language detection

---

## Additional Notes

### Installing CUDA for Windows/Linux with GPU
If using CUDA on an NVIDIA GPU, make sure to:
- Install and update the NVIDIA drivers.
- Optionally install the CUDA Toolkit. [Download from NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### Optimization for macOS
PyTorch leverages MPS (Metal Performance Shaders) to accelerate computations on Apple Silicon. No additional configuration is required.

---

## Development and Contributions

If you'd like to contribute, ensure you test the project on multiple platforms and configurations (Windows, macOS, GPU, CPU).

---

## Authors

Project developed by [Your Name/Team].
