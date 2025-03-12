import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import yt_dlp
import re
import os
import shutil

# Path configuration
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
WHISPER_CPP_DIR = PROJECT_ROOT / "whisper.cpp"
WHISPER_CLI_PATH = WHISPER_CPP_DIR / "build/bin/whisper-cli"
MODELS_DIR = WHISPER_CPP_DIR / "models"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_progress_hook(d: dict) -> None:
    """Hook to display download progress."""
    if d['status'] == 'downloading':
        logger.info(f"Downloading: {d.get('_percent_str', 'N/A')}")

def sanitize_filename(filename: str) -> str:
    """Cleans a filename to make it safe for the file system."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F：\s]', '_', filename)
    return re.sub(r'_{2,}', '_', sanitized).strip('_')

def download_audio(input_url: str, output_dir: Path) -> Tuple[List[Path], List[str]]:
    """
    Downloads audio from a YouTube URL.
    
    Args:
        input_url: YouTube URL
        output_dir: Output directory
        
    Returns:
        Tuple containing the list of downloaded audio files and the list of failed URLs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / 'TEMP'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'progress_hooks': [download_progress_hook],
        'ignoreerrors': True,
    }

    audio_files = []
    failed_urls = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(input_url, download=False)
        entries = info_dict.get('entries', [info_dict]) if info_dict else []

        for entry in entries:
            if not entry:
                continue
            
            try:
                temp_path = output_dir / "TEMP.wav"
                ydl.download([entry['webpage_url']])
                
                if temp_path.exists():
                    sanitized_name = sanitize_filename(f"{entry['title']}.wav")
                    audio_path = output_dir / sanitized_name
                    os.rename(temp_path, audio_path)
                    audio_files.append(audio_path)
                    logger.info(f"Audio downloaded: {audio_path}")
                else:
                    failed_urls.append(entry['webpage_url'])
            except Exception as e:
                logger.error(f"Download failed: {e}")
                failed_urls.append(entry.get('webpage_url', 'unknown'))

    return audio_files, failed_urls

def check_whisper_cli() -> bool:
    """Checks if whisper-cli exists and is executable."""
    if not WHISPER_CLI_PATH.exists():
        logger.error(f"whisper-cli not found at {WHISPER_CLI_PATH}")
        logger.error("Please compile whisper.cpp first")
        return False
    return True

def check_model(model_name: str) -> Path:
    """
    Checks if the model exists, otherwise downloads it.
    
    Args:
        model_name: Model name (e.g., large-v3-turbo)
        
    Returns:
        Path to the model
    """
    model_path = MODELS_DIR / f"ggml-{model_name}.bin"
    
    if not model_path.exists():
        logger.info(f"Model {model_name} not found. Downloading...")
        
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
        download_script = WHISPER_CPP_DIR / "models/download-ggml-model.sh"
        
        if not download_script.exists():
            logger.error(f"Download script not found at {download_script}")
            raise FileNotFoundError(f"Download script not found")
        
        try:
            subprocess.run([str(download_script), model_name], 
                          cwd=WHISPER_CPP_DIR, 
                          check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Model download failed: {e}")
            raise
    
    return model_path

def transcribe_audio(audio_path: Path, output_dir: Path, model_name: str, language: Optional[str] = None) -> Path:
    """
    Transcribes an audio file using whisper-cli.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Output directory
        model_name: Model name
        language: Language code (optional). If None, language will be automatically detected.
        
    Returns:
        Path to the transcribed file
    """
    # Check prerequisites
    if not check_whisper_cli():
        raise RuntimeError("whisper-cli not available")
    
    model_path = check_model(model_name)
    
    # Prepare output paths
    base_filename = audio_path.stem
    transcript_path = output_dir / f"{base_filename}_transcription.txt"
    
    # Convert all paths to absolute paths
    audio_path_abs = audio_path.absolute()
    model_path_abs = model_path.absolute()
    
    # Build the command
    command = [
        str(WHISPER_CLI_PATH),
        "-m", str(model_path_abs),
        "-f", str(audio_path_abs),
        "-otxt",
        "-of", str(output_dir.absolute() / base_filename)  # Specify output path
    ]
    
    if language:
        command.extend(["-l", language])
        logger.info(f"Using specified language: {language}")
    else:
        logger.info("No language specified, automatic detection will be used")
    
    logger.info(f"Executing command: {' '.join(command)}")
    
    try:
        # Execute whisper-cli
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # The output file should now be directly in the output directory
        output_filename = f"{base_filename}.txt"
        generated_path = output_dir.absolute() / output_filename
        
        if generated_path.exists():
            # Rename with _transcription suffix
            os.rename(str(generated_path), str(transcript_path))
        else:
            # Create from stdout if the file doesn't exist
            logger.warning(f"Output file {generated_path} not found, using stdout")
            transcript_path.write_text(result.stdout)
        
        # Extract detected language, if available in the output
        detected_lang = None
        for line in result.stdout.splitlines():
            if "Detected language" in line:
                detected_lang = line.split(":")[-1].strip()
                break
        
        if not language and detected_lang:
            logger.info(f"Automatically detected language: {detected_lang}")
        
        logger.info(f"Transcription completed: {transcript_path}")
        return transcript_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during transcription: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Transcription error: {e}")

def main(input_url: str, output_dir: str, model_name: str, language: Optional[str] = None, keep_audio: bool = False) -> None:
    """
    Main function of the script.
    
    Args:
        input_url: YouTube URL
        output_dir: Output directory
        model_name: Model name
        language: Language code (optional)
        keep_audio: Keep audio files after transcription
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Downloading audio from: {input_url}")
    audio_files, failed_urls = download_audio(input_url, output_path)
    
    if not audio_files:
        logger.error("No audio files downloaded")
        return
    
    if failed_urls:
        logger.warning(f"Failed URLs: {', '.join(failed_urls)}")
    
    for audio_file in audio_files:
        try:
            transcript_path = transcribe_audio(audio_file, output_path, model_name, language)
            logger.info(f"Transcription successful: {transcript_path}")
        except Exception as e:
            logger.error(f"Transcription failed for {audio_file}: {e}")
        finally:
            if not keep_audio and audio_file.exists():
                logger.info(f"Deleting audio file: {audio_file}")
                audio_file.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YouTube transcription with whisper-cli")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-m", "--model", default="large-v3-turbo", help="Model to use (large-v3-turbo, base, etc.)")
    parser.add_argument("-l", "--lang", help="Target language (if not specified, language will be automatically detected)")
    parser.add_argument("-k", "--keep-audio", action="store_true", help="Keep audio files after transcription")
    
    args = parser.parse_args()
    
    main(args.url, args.output, args.model, args.lang, args.keep_audio)