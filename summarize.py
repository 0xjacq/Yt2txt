#!/usr/bin/env python3
import argparse
import json
import logging
import os
import requests
from pathlib import Path
from typing import Optional, Dict, Any

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_transcript(transcript_path: Path) -> str:
    """
    Reads the content of a transcript file.
    
    Args:
        transcript_path: Path to the transcript file
        
    Returns:
        Content of the transcript file
    """
    logger.info(f"Reading transcript file: {transcript_path}")
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"File read successfully ({len(content)} characters)")
        return content
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise

def generate_summary(transcript: str, api_key: str, model: str = "anthropic/claude-3-haiku", max_tokens: int = 1000) -> Dict[str, Any]:
    """
    Generates a summary from a transcript using OpenRouter.
    
    Args:
        transcript: Transcript content
        api_key: OpenRouter API key
        model: Model to use for generation
        max_tokens: Maximum number of tokens for the response
        
    Returns:
        OpenRouter API response
    """
    logger.info(f"Generating summary with model {model}")
    
    # Prepare the request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost"  # Required by OpenRouter
    }
    
    # Build the prompt
    system_prompt = """You are an assistant specialized in creating clear, concise, and informative summaries. 
Your task is to summarize the provided transcript while preserving the main information, 
key points, and content structure. Your summary should be well-organized, easy to read, 
and capture the essence of the original document."""
    
    user_prompt = f"""Here is a transcript to summarize:

{transcript}

Generate a structured summary that:
1. Captures the main topics and ideas
2. Highlights the essential points
3. Maintains the chronological order of the topics covered
4. Is easy to read and understand
5. Includes a short introduction and conclusion if relevant"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens
    }
    
    # Send the request
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling the OpenRouter API: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Details: {e.response.text}")
        raise

def save_summary(summary: str, output_path: Path) -> Path:
    """
    Saves the summary to a file.
    
    Args:
        summary: Summary to save
        output_path: Output file path
        
    Returns:
        Path to the saved file
    """
    logger.info(f"Saving summary to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary saved successfully ({len(summary)} characters)")
        return output_path
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
        raise

def get_model_short_name(model: str) -> str:
    """
    Extracts a short name from the model identifier.
    
    Args:
        model: Full model identifier (e.g., "anthropic/claude-3-haiku")
        
    Returns:
        Short model name (e.g., "claude-3-haiku")
    """
    # Extract the model name after the provider
    if "/" in model:
        return model.split("/")[-1]
    return model

def main(transcript_path: str, api_key: Optional[str] = None, model: str = "anthropic/claude-3-haiku", output_path: Optional[str] = None) -> None:
    """
    Main function.
    
    Args:
        transcript_path: Path to the transcript file
        api_key: OpenRouter API key (if not provided, looks for OPENROUTER_API_KEY environment variable)
        model: Model to use
        output_path: Output file path (if not provided, uses the transcript filename with "_summary" added)
    """
    # Check API key
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided and not found in environment variables (OPENROUTER_API_KEY)")
    
    # Convert paths to Path objects
    transcript_file = Path(transcript_path)
    
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    # Get a short version of the model name for the filename
    model_short_name = get_model_short_name(model)
    
    # Define output path if not provided
    if not output_path:
        output_file = transcript_file.parent / f"{transcript_file.stem}_summary_{model_short_name}.txt"
    else:
        # If output path is provided but doesn't include model name, add it before the extension
        output_file = Path(output_path)
        if model_short_name not in output_file.stem:
            output_file = output_file.with_stem(f"{output_file.stem}_{model_short_name}")
    
    # Read the transcript
    transcript = read_transcript(transcript_file)
    
    # Generate the summary
    response = generate_summary(transcript, api_key, model)
    
    # Extract the summary from the response
    if 'choices' in response and len(response['choices']) > 0:
        summary = response['choices'][0]['message']['content']
        
        # Save the summary
        save_summary(summary, output_file)
        
        logger.info("Summary generated and saved successfully!")
        logger.info(f"Model used: {model}")
        logger.info(f"Output file: {output_file}")
    else:
        logger.error(f"Unexpected API response: {response}")
        raise RuntimeError("Unexpected response format from OpenRouter API")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a summary from a transcript using OpenRouter")
    parser.add_argument("transcript", help="Path to the transcript file")
    parser.add_argument("-k", "--api-key", help="OpenRouter API key (if not provided, uses the OPENROUTER_API_KEY environment variable)")
    parser.add_argument("-m", "--model", default="anthropic/claude-3-haiku", help="Model to use (default: anthropic/claude-3-haiku)")
    parser.add_argument("-o", "--output", help="Output file path (if not provided, uses the transcript filename with '_summary_[model]' added)")
    
    args = parser.parse_args()
    
    main(args.transcript, args.api_key, args.model, args.output) 