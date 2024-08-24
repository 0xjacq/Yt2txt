# Yt2Txt - YouTube Audio Transcriber

This script downloads and transcribes audio from a YouTube video using Whisper by OpenAI and Cuda.

Very accurate and fast on a good GPU.

## Requirements

- Python 3.x
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
