# Yt2Txt - YouTube Audio Transcriber

This script downloads and transcribes audio from a YouTube video using Whisper by OpenAI and Cuda.

Very accurate and fast on a good GPU.

## Requirements

- Python 3.x
- CUDA 12.4

## Dependencies 

- yt-dlp
- faster-whisper
- torch

## Installation

Install the required dependencies:

```sh
pip install -r requirements.txt
```
## Usage

```sh
yt2txt.py [-h] [-o OUTPUT_DIR] [-m WHISPER_MODEL] youtube_url
```

Compatible with youtube playlist links (be aware that youtube may blcok you if the playlist is too long)

## TODO

