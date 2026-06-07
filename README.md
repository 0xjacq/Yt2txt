# Yt2txt

`yt2txt` turns YouTube videos or local `.wav` files into plain-text transcripts.
For YouTube inputs it now prefers original-language text tracks before running
local ASR:

1. creator-provided subtitles
2. YouTube automatic captions
3. local `mlx-qwen3-asr` fallback

The emitted transcript contract stays the same for downstream callers:

- per-video `*_transcription.txt`
- latest transcript copied to `output.txt`

## Requirements

- Python 3.10+
- `ffmpeg`
- `yt-dlp`
- `mlx-qwen3-asr`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

Basic usage:

```bash
python yt2txt.py "https://www.youtube.com/watch?v=VIDEO_ID" --output output
```

Caption-first behavior is the default:

```bash
python yt2txt.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output output \
  --transcript-source prefer-captions
```

Available source modes:

- `prefer-captions`
  - default
  - manual subtitles, then automatic captions, then ASR
- `captions-only`
  - fail if no usable original/default-language caption track exists
- `asr-only`
  - skip captions and always run the local ASR pipeline

Preserve human-readable caption markers when using caption tracks:

```bash
python yt2txt.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output output \
  --markers
```

Useful options:

- `--lang`
  - preferred original-language caption key or ASR language override
- `--draft-model`
  - speculative decoding draft model for `mlx-qwen3-asr`
- `--keep-audio`
  - keep downloaded `.wav` files after ASR
- `--max-videos`
  - limit playlist/channel processing

## Local Audio

Single files and directories of `.wav` inputs remain ASR-only:

```bash
python yt2txt.py /path/to/audio.wav --output output
python yt2txt.py /path/to/wavs --output output
```

`captions-only` is rejected for local audio inputs.

## Python API

The existing `main(...)` entry point remains available and now accepts two extra
optional parameters:

```python
from yt2txt import main

paths = main(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    "output",
    "Qwen/Qwen3-ASR-0.6B",
    transcript_source="prefer-captions",
    markers=False,
)
```

## Notes

- Caption downloads are normalized into plain transcript text.
- VTT timing, cue numbering, and markup are always removed.
- By default, caption artifacts like `>>` and `[music]` are removed.
- `--markers` preserves those human-readable cues while still removing VTT
  structure.
