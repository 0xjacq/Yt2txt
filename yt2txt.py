import logging
import argparse
from pathlib import Path
import yt_dlp
from faster_whisper import WhisperModel
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_progress_hook(d):
    if d['status'] == 'downloading':
        percent = d['_percent_str']
        speed = d['_speed_str']
        eta = d['_eta_str']
        logger.info(f"Downloading: {percent} complete (Speed: {speed}, ETA: {eta})")
    elif d['status'] == 'finished':
        logger.info("Download completed. Starting audio extraction...")

def is_cuda_available():
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        logger.warning("CUDA is not available. Falling back to CPU.")
        return False

def download_audio(input_url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'progress_hooks': [download_progress_hook],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(input_url, download=True)
        return Path(ydl.prepare_filename(info_dict)).with_suffix('.wav')

def transcribe_audio(audio_file_path: Path, output_dir: Path, model_size: str) -> Path:
    logger.info("Starting transcription...")
    device = "cuda" if is_cuda_available() else "cpu"
    logger.info(f"Using device: {device}")
    model = WhisperModel(model_size, device=device, compute_type="float16")
    segments, info = model.transcribe(str(audio_file_path), beam_size=5)

    logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    text = ""
    for segment in segments:
        logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        text += segment.text + " "

    transcripted_file_path = output_dir / f"{audio_file_path.stem}_transcribed.txt"
    transcripted_file_path.write_text(text.strip(), encoding="utf-8")
    logger.info(f"Transcription saved to {transcripted_file_path}")
    return transcripted_file_path

def cleanup(audio_file_path: Path):
    if audio_file_path and audio_file_path.exists():
        audio_file_path.unlink()
        logger.info(f"Removed temporary audio file: {audio_file_path}")

def main(input_url: str, output_dir: str, model_size: str):
    output_path = Path(output_dir)
    try:
        audio_file_path = download_audio(input_url, output_path)
        transcribe_audio(audio_file_path, output_path, model_size)
    finally:
        # Cleanup the audio file regardless of any exceptions
        cleanup(audio_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and transcribe audio from a YouTube video.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", default="output", help="Output directory for transcription")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model size to use")
    args = parser.parse_args()

    main(args.url, args.output, args.model)
