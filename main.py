#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple

# Import functions from other scripts
from yt2txt import main as yt2txt_main, download_audio
from summarize import main as summarize_main

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_deterministic_filenames(youtube_url: str, output_dir: Path, model_name: str = "") -> Tuple[Path, Path]:
    """
    Generate deterministic filenames based on YouTube URL and video title.
    
    Args:
        youtube_url: YouTube URL
        output_dir: Output directory
        model_name: Model name to include in the summary filename
        
    Returns:
        Tuple with paths for transcript file and summary file
    """
    # Generate a short but deterministic hash of the URL
    url_hash = hashlib.md5(youtube_url.encode()).hexdigest()[:8]
    
    # Get the YouTube video title
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', '')
            
            # Clean the title to make it usable as a filename
            from yt2txt import sanitize_filename
            clean_title = sanitize_filename(video_title)
            
            # Limit the title length (max 50 characters)
            if len(clean_title) > 50:
                clean_title = clean_title[:47] + "..."
    except Exception as e:
        logger.warning(f"Unable to retrieve video title: {e}")
        clean_title = f"video_{url_hash}"
    
    # Extract a short version of the model name if provided
    model_suffix = ""
    if model_name:
        # Extract the model name after the provider
        if "/" in model_name:
            model_short_name = model_name.split("/")[-1]
        else:
            model_short_name = model_name
        model_suffix = f"_{model_short_name}"
    
    # Create filenames based on title and hash
    transcript_file = output_dir / f"{clean_title}_transcript.txt"
    summary_file = output_dir / f"{clean_title}_summary{model_suffix}.txt"
    
    # If files already exist with different content (name collision),
    # add the hash to ensure uniqueness
    if transcript_file.exists() or summary_file.exists():
        transcript_file = output_dir / f"{clean_title}_{url_hash}_transcript.txt"
        summary_file = output_dir / f"{clean_title}_{url_hash}_summary{model_suffix}.txt"
    
    # Log the filenames for debugging
    logger.info(f"Determined transcript file: {transcript_file}")
    logger.info(f"Determined summary file: {summary_file}")
    
    return transcript_file, summary_file

def process_video(
    youtube_url: str,
    output_dir: str = "output",
    model_name: str = "",
    custom_prompt: Optional[str] = None,
    whisper_model: str = "large-v3-turbo",
    language: Optional[str] = None,
    keep_audio: bool = False,
    api_key: Optional[str] = None,
    force_regenerate: bool = False,
    verbose_transcription: bool = True
) -> Tuple[Path, Path]:
    """
    Complete processing of a YouTube video: download, transcription, and summary.
    If no model_name is provided, only transcription will be performed.
    
    Args:
        youtube_url: YouTube URL
        output_dir: Output directory
        model_name: LLM model to use for summarization (if empty, no summary is generated)
        custom_prompt: Custom prompt for summarization
        whisper_model: Whisper model to use for transcription
        language: Language code for transcription
        keep_audio: Keep audio files after transcription
        api_key: OpenRouter API key
        force_regenerate: Force regeneration even if files already exist
        verbose_transcription: Display detailed transcription logs
        
    Returns:
        Tuple containing paths to transcript and summary files (summary may be None)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate deterministic filenames
    transcript_file, summary_file = get_deterministic_filenames(youtube_url, output_path, model_name)
    
    # Step 1: Transcription with yt2txt.py (if necessary)
    if not transcript_file.exists() or force_regenerate:
        logger.info("=" * 60)
        logger.info("=== STEP 1: DOWNLOAD AND TRANSCRIPTION ===")
        logger.info("=" * 60)
        logger.info(f"YouTube URL: {youtube_url}")
        logger.info(f"Whisper Model: {whisper_model}")
        if language:
            logger.info(f"Specified Language: {language}")
        else:
            logger.info("Language: Automatic detection")
        logger.info(f"Destination: {output_path}")
        logger.info("-" * 60)
        
        try:
            # Phase 1.1: Download
            logger.info("[Phase 1.1] Starting audio download...")
            import time
            start_time = time.time()
            
            # Download audio
            audio_files, failed_urls = download_audio(youtube_url, output_path)
            
            if not audio_files:
                logger.error("❌ Failure: No audio file downloaded. Stopping process.")
                sys.exit(1)
            
            download_time = time.time() - start_time
            logger.info(f"✓ Download completed in {download_time:.2f} seconds")
            
            # Use the transcription function but write to our deterministic file
            audio_file = audio_files[0]  # Use the first downloaded audio file
            
            # Display audio file information
            audio_size_mb = audio_file.stat().st_size / (1024 * 1024)
            logger.info(f"[INFO] Audio file: {audio_file.name} ({audio_size_mb:.2f} MB)")
            
            try:
                # Phase 1.2: Transcription
                logger.info("-" * 60)
                logger.info("[Phase 1.2] Starting transcription...")
                logger.info(f"[INFO] Using Whisper model: {whisper_model}")
                
                transcription_start = time.time()
                
                # Direct call to whisper-cli with real-time output capture
                if verbose_transcription:
                    # Use subprocess with real-time output capture
                    import subprocess
                    import threading
                    from yt2txt import WHISPER_CLI_PATH, check_whisper_cli, check_model
                    
                    if not check_whisper_cli():
                        raise RuntimeError("whisper-cli not available")
                    
                    model_path = check_model(whisper_model)
                    
                    # Convert all paths to absolute paths
                    audio_path_abs = audio_file.absolute()
                    model_path_abs = model_path.absolute()
                    output_base = output_path.absolute() / audio_file.stem
                    
                    # Build the command
                    command = [
                        str(WHISPER_CLI_PATH),
                        "-m", str(model_path_abs),
                        "-f", str(audio_path_abs),
                        "-otxt",
                        "-of", str(output_base),
                        "-pc",  # Print segments as they are transcribed
                    ]
                    
                    if language:
                        command.extend(["-l", language])
                    
                    logger.info(f"Executing command: {' '.join(command)}")
                    logger.info("-" * 60)
                    logger.info("TRANSCRIPTION IN PROGRESS (real-time segments):")
                    logger.info("-" * 60)
                    
                    # Execute whisper-cli with real-time output capture
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )
                    
                    # Function to read and process output in real-time
                    def process_output(pipe, is_error=False):
                        transcript_content = []
                        segment_count = 0
                        
                        for line in iter(pipe.readline, ''):
                            line = line.strip()
                            if not line:
                                continue
                                
                            if is_error:
                                logger.error(f"[whisper-cli] {line}")
                            else:
                                # Filter lines containing transcribed text
                                if "[" in line and "]" in line and not line.startswith("whisper_"):
                                    segment_count += 1
                                    logger.info(f"Segment {segment_count}: {line}")
                                    # Extract text without timestamps for the final file
                                    if "]" in line:
                                        text_part = line.split("]", 1)[1].strip()
                                        transcript_content.append(text_part)
                                else:
                                    # Print other output lines as info
                                    logger.info(f"[whisper-cli] {line}")
                        
                        return " ".join(transcript_content)
                    
                    # Create threads to handle stdout and stderr
                    stdout_thread = threading.Thread(target=lambda: process_output(process.stdout))
                    stderr_thread = threading.Thread(target=lambda: process_output(process.stderr, True))
                    
                    # Start threads
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    # Wait for process to complete
                    return_code = process.wait()
                    
                    # Wait for threads to complete
                    stdout_thread.join()
                    stderr_thread.join()
                    
                    # Check if the process completed successfully
                    if return_code != 0:
                        logger.error(f"Error during transcription: return code {return_code}")
                        raise RuntimeError(f"Transcription error: code {return_code}")
                    
                    # Read the output file generated by whisper-cli
                    temp_output = Path(f"{output_base}.txt")
                    if temp_output.exists():
                        with open(temp_output, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Write content to our deterministic file
                        with open(transcript_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        # Clean up temporary file
                        os.remove(temp_output)
                    else:
                        logger.warning(f"Expected output file {temp_output} not found")
                        # Try to reconstruct from captured output
                        logger.info("Attempting to reconstruct transcript from captured output")
                        # We'll use the content from the output file in the next steps
                    
                    transcript_path = transcript_file
                    
                else:
                    # Use the original method without detailed logs
                    from yt2txt import transcribe_audio
                    transcript_path = transcribe_audio(audio_file, output_path, whisper_model, language)
                    
                    # Rename the file with our deterministic name
                    if transcript_path.exists() and transcript_path != transcript_file:
                        logger.info("[Phase 1.3] Renaming transcription file...")
                        with open(transcript_path, 'r', encoding='utf-8') as src:
                            content = src.read()
                            
                        with open(transcript_file, 'w', encoding='utf-8') as dst:
                            dst.write(content)
                        
                        os.remove(transcript_path)
                        transcript_path = transcript_file
                
                transcription_time = time.time() - transcription_start
                logger.info("-" * 60)
                logger.info(f"✓ Transcription completed in {transcription_time:.2f} seconds")
                
                # Display transcription statistics
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    word_count = len(content.split())
                    char_count = len(content)
                
                logger.info(f"✓ Transcription file: {transcript_file}")
                logger.info(f"[INFO] Transcription statistics: {word_count} words, {char_count} characters")
                
                total_time = time.time() - start_time
                logger.info("-" * 60)
                logger.info(f"✓ Total processing time: {total_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"❌ Transcription failed: {e}")
                raise
            finally:
                # Clean up audio files if necessary
                if not keep_audio and audio_file.exists():
                    logger.info("[Phase 1.4] Cleaning up temporary files...")
                    logger.info(f"Deleting audio file: {audio_file}")
                    audio_file.unlink()
                    logger.info("✓ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during transcription: {e}")
            sys.exit(1)
    else:
        logger.info(f"Transcription file already exists: {transcript_file}")
        logger.info("Transcription step skipped.")
    
    # Step 2: Summary generation with summarize.py (if necessary)
    # Skip summary generation if no model is specified
    if model_name and (not summary_file.exists() or force_regenerate):
        logger.info("\n" + "=" * 60)
        logger.info("=== STEP 2: SUMMARY GENERATION ===")
        logger.info("=" * 60)
        
        try:
            logger.info(f"Source transcription: {transcript_file.name}")
            logger.info(f"LLM Model: {model_name}")
            
            # Statistics on the transcription file
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
                word_count = len(content.split())
                char_count = len(content)
                logger.info(f"[INFO] Transcription statistics: {word_count} words, {char_count} characters")
            
            # If a custom prompt is provided, pass it via an environment variable
            if custom_prompt:
                os.environ["SUMMARY_CUSTOM_PROMPT"] = custom_prompt
                logger.info(f"[INFO] Using custom prompt: {custom_prompt[:50]}...")
            else:
                # Make sure the variable is cleared if it existed from a previous call
                os.environ.pop("SUMMARY_CUSTOM_PROMPT", None)
                logger.info("[INFO] Using default prompt")
            
            logger.info("-" * 60)
            logger.info("Starting summary generation...")
            import time
            start_time = time.time()
            
            # Call the main function of summarize.py with our custom filename
            summarize_main(str(transcript_file), api_key, model_name, str(summary_file))
            
            # Statistics on the summary
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_content = f.read()
                summary_word_count = len(summary_content.split())
                summary_char_count = len(summary_content)
            
            summarize_time = time.time() - start_time
            logger.info(f"✓ Summary generated in {summarize_time:.2f} seconds")
            logger.info(f"✓ Summary saved: {summary_file}")
            logger.info(f"[INFO] Summary statistics: {summary_word_count} words, {summary_char_count} characters")
            logger.info(f"[INFO] Compression rate: {(1 - summary_word_count/word_count) * 100:.1f}%")
                
        except Exception as e:
            logger.error(f"❌ Error during summary generation: {e}")
            raise
    elif model_name:
        logger.info(f"Summary file already exists: {summary_file}")
        logger.info("Summary generation step skipped.")
    else:
        logger.info("\nNo LLM model specified. Skipping summary generation.")
        summary_file = None
    
    return transcript_file, summary_file

if __name__ == "__main__":
    # Modify summarize.py to support custom prompts
    # We need to intercept and modify the module before importing its functions
    import importlib.util
    import sys
    from types import ModuleType

    spec = importlib.util.spec_from_file_location("summarize", "summarize.py")
    summarize = importlib.util.module_from_spec(spec)
    sys.modules["summarize"] = summarize
    spec.loader.exec_module(summarize)

    # Patch the generate_summary function to support custom prompts
    original_generate_summary = summarize.generate_summary
    
    def patched_generate_summary(transcript, api_key, model="anthropic/claude-3-haiku", max_tokens=1000):
        """Modified version that supports custom prompts."""
        custom_prompt = os.environ.get("SUMMARY_CUSTOM_PROMPT")
        
        if custom_prompt:
            # Modify the payload to include the custom prompt
            logger.info("Using custom prompt for summary")
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an assistant specialized in creating summaries."},
                    {"role": "user", "content": f"{custom_prompt}\n\nHere is the transcription to summarize:\n\n{transcript}"}
                ],
                "max_tokens": max_tokens
            }
            
            import requests
            import json
            
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling the OpenRouter API: {e}")
                if hasattr(e, 'response') and e.response:
                    logger.error(f"Details: {e.response.text}")
                raise
        else:
            # Use the original function if no custom prompt
            return original_generate_summary(transcript, api_key, model, max_tokens)
    
    # Replace the original function with our version
    summarize.generate_summary = patched_generate_summary
    
    # Now, parse arguments
    parser = argparse.ArgumentParser(description="Complete YouTube video processing: download, transcription, and summary")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("-m", "--llm-model", default="", help="LLM model to use for summarization (e.g., openai/gpt-3.5-turbo). If not specified, only transcription will be performed.")
    parser.add_argument("-p", "--prompt", help="Custom prompt to guide summary generation")
    parser.add_argument("-w", "--whisper-model", default="large-v3-turbo", help="Whisper model to use for transcription (default: large-v3-turbo)")
    parser.add_argument("-l", "--lang", help="Language code for transcription (automatic detection by default)")
    parser.add_argument("-k", "--keep-audio", action="store_true", help="Keep audio files after transcription")
    parser.add_argument("-a", "--api-key", help="OpenRouter API key (if not provided, uses the OPENROUTER_API_KEY environment variable)")
    parser.add_argument("-f", "--force", action="store_true", help="Force regeneration even if files already exist")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display detailed transcription logs")
    
    args = parser.parse_args()
    
    # Execute the complete process
    logger.info(f"Starting video processing: {args.url}")
    transcript_file, summary_file = process_video(
        args.url,
        args.output,
        args.llm_model,
        args.prompt,
        args.whisper_model,
        args.lang,
        args.keep_audio,
        args.api_key,
        args.force,
        args.verbose
    )
    
    # Display result
    logger.info("\n=== PROCESSING RESULT ===")
    logger.info(f"Transcription file: {transcript_file}")
    if summary_file:
        logger.info(f"Summary file: {summary_file}")
    else:
        logger.info("No summary generated (no LLM model specified)")
    
    logger.info("\nProcessing completed successfully!") 