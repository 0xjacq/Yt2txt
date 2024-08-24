import logging
import argparse
from pathlib import Path
from typing import List
import yt_dlp
from faster_whisper import WhisperModel
from torch.cuda import is_available as cuda_is_available
from torch.cuda import get_device_name as cuda_get_device_name
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_progress_hook(d: dict) -> None:
    """
    Progress hook to log download progress in yt-dlp.

    :param d: Dictionary containing download status information
    """
    if d['status'] == 'downloading':
        percent = d['_percent_str']
        speed = d['_speed_str']
        eta = d['_eta_str']
        logger.info(f"Downloading: {percent} complete (Speed: {speed}, ETA: {eta})")
    elif d['status'] == 'finished':
        logger.info("Download completed. Starting audio extraction...")

def is_cuda_available() -> bool:
    """
    Check if CUDA is available and log the GPU information if available.

    :return: True if CUDA is available, False otherwise
    """
    if cuda_is_available():
        logger.info(f"CUDA is available. GPU: {cuda_get_device_name(0)}")
        return True
    else:
        logger.warning("CUDA is not available. Falling back to CPU.")
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing characters that are not allowed in filenames.

    :param filename: Original filename
    :return: Sanitized filename
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)
    return sanitized

def download_audio(input_url: str, output_dir: Path, sleep_interval: int) -> List[Path]:
    """
    Download audio from a YouTube video or playlist and convert it to WAV format.

    :param input_url: The URL of the YouTube video or playlist
    :param output_dir: Directory where the audio files will be saved
    :param sleep_interval: Time interval (in seconds) to sleep between each download
    :return: List of paths to the downloaded audio files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(title)s_%(id)s.%(ext)s'),  # Unique filenames with video ID
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'progress_hooks': [download_progress_hook],
        'sleep_interval': sleep_interval,  # Add sleep interval between downloads
        'ignoreerrors': True,  # Ignore errors and continue downloading other videos
    }

    # Initialize yt-dlp with options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Get metadata without downloading
        info_dict = ydl.extract_info(input_url, download=False)

        audio_files = []  # List to store paths to downloaded audio files
        failed_urls = []  # List to store URLs of videos that couldn't be downloaded

        # Check if we are dealing with a playlist or a single video
        if 'entries' in info_dict:
            # Playlist case
            for entry in info_dict['entries']:
                if entry is None:
                    continue  # Skip entries that couldn't be processed
                temp_file_name = f"{entry['title']}_{entry['id']}.wav"
                temp_file_path = output_dir / temp_file_name
                sanitized_file_name = sanitize_filename(temp_file_name)
                sanitized_file_path = output_dir / sanitized_file_name
                try:
                    ydl.download([entry['webpage_url']])  # Download the video
                    if temp_file_path.exists():
                        os.rename(temp_file_path, sanitized_file_path)
                        audio_files.append(sanitized_file_path)
                    else:
                        logger.error(f"File not found after download: {temp_file_path}")
                        failed_urls.append(entry['webpage_url'])
                except Exception as e:
                    logger.warning(f"Failed to download {entry['webpage_url']}: {e}")
                    failed_urls.append(entry['webpage_url'])
        else:
            # Single video case
            temp_file_name = f"{info_dict['title']}_{info_dict['id']}.wav"
            temp_file_path = output_dir / temp_file_name
            sanitized_file_name = sanitize_filename(temp_file_name)
            sanitized_file_path = output_dir / sanitized_file_name
            try:
                ydl.download([input_url])  # Download the video
                if temp_file_path.exists():
                    os.rename(temp_file_path, sanitized_file_path)
                    audio_files.append(sanitized_file_path)
                else:
                    logger.error(f"File not found after download: {temp_file_path}")
                    failed_urls.append(input_url)
            except Exception as e:
                logger.warning(f"Failed to download {input_url}: {e}")
                failed_urls.append(input_url)

        return audio_files, failed_urls

def transcribe_audio(audio_file_path: Path, output_dir: Path, model_size: str) -> Path:
    """
    Transcribe audio to text using the Whisper model.

    :param audio_file_path: Path to the audio file to transcribe
    :param output_dir: Directory where the transcription will be saved
    :param model_size: Whisper model size to use for transcription
    :return: Path to the transcription file
    """
    logger.info("Starting transcription...")

    # Choose the appropriate device (GPU or CPU)
    device = "cuda" if is_cuda_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Whisper model
    model = WhisperModel(model_size, device=device, compute_type="float16")

    # Perform transcription
    segments, info = model.transcribe(str(audio_file_path), beam_size=5)

    logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    # Build the transcription text
    text = ""
    for segment in segments:
        logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        text += segment.text + " "

    # Save the transcription to a file
    transcripted_file_name = sanitize_filename(f"{audio_file_path.stem}_transcribed.txt")
    transcripted_file_path = output_dir / transcripted_file_name
    transcripted_file_path.write_text(text.strip(), encoding="utf-8")
    logger.info(f"Transcription saved to {transcripted_file_path}")

    return transcripted_file_path

def cleanup(audio_file_path: Path) -> None:
    """
    Remove the temporary audio file after transcription is done.

    :param audio_file_path: Path to the audio file to be deleted
    """
    if audio_file_path.exists():
        audio_file_path.unlink()
        logger.info(f"Removed temporary audio file: {audio_file_path}")

def main(input_url: str, output_dir: str, model_size: str, sleep_interval: int) -> None:
    """
    Main function to download, transcribe, and clean up audio files from a YouTube video or playlist.

    :param input_url: The URL of the YouTube video or playlist
    :param output_dir: Directory where output files will be saved
    :param model_size: Whisper model size to use for transcription
    :param sleep_interval: Time interval (in seconds) to sleep between each download
    """
    output_path = Path(output_dir)
    try:
        # Download the audio files
        audio_files, failed_urls = download_audio(input_url, output_path, sleep_interval)

        # Transcribe each audio file and clean up after transcription
        for audio_file_path in audio_files:
            try:
                transcribe_audio(audio_file_path, output_path, model_size)
            finally:
                cleanup(audio_file_path)

        # Log the URLs of videos that could not be downloaded
        if failed_urls:
            logger.warning("The following videos could not be downloaded:")
            for url in failed_urls:
                logger.warning(url)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("All files processed.")

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Download and transcribe audio from a YouTube video or playlist.")
    parser.add_argument("url", help="YouTube video or playlist URL")
    parser.add_argument("-o", "--output", default="output", help="Output directory for transcription")
    parser.add_argument("-m", "--model", default="large-v3", help="Whisper model size to use")
    parser.add_argument("-s", "--sleep-interval", type=int, default=1, help="Sleep interval between downloads (in seconds)")
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.url, args.output, args.model, args.sleep_interval)
