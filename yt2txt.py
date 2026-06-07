from __future__ import annotations

import argparse
import html
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yt_dlp


PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}"
)
CUE_NUMBER_RE = re.compile(r"^\d+$")
VTT_TAG_RE = re.compile(r"</?[^>]+>")
VOICE_MARKER_RE = re.compile(r"^\s*>>\s*")
STAGE_DIRECTION_KEYWORDS = [
    "applause",
    "audience",
    "beep",
    "bleep",
    "cheering",
    "cheers",
    "clears throat",
    "gasps",
    "inaudible",
    "instrumental",
    "instrumental music",
    "laughter",
    "laughing",
    "music",
    "phone ringing",
    "ringing",
    "silence",
    "singing",
    "sighs",
]
STAGE_DIRECTION_RE = re.compile(
    r"\[(?:"
    + "|".join(re.escape(keyword) for keyword in STAGE_DIRECTION_KEYWORDS)
    + r")\]",
    re.IGNORECASE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptionTrackSelection:
    source: str
    language_key: str


def download_progress_hook(download_state: dict) -> None:
    """Hook to display download progress."""
    if download_state["status"] == "downloading":
        logger.info("Downloading: %s", download_state.get("_percent_str", "N/A"))


def sanitize_filename(filename: str) -> str:
    """Clean a filename so it is safe for the file system."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F：\s]', "_", filename)
    return re.sub(r"_{2,}", "_", sanitized).strip("_")


def write_output_copy(transcript_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(transcript_path, output_dir / "output.txt")


def normalize_language_value(language: Optional[str]) -> List[str]:
    if not language:
        return []

    normalized = language.strip().lower().replace("_", "-")
    candidates: List[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if normalized.endswith("-orig"):
        base = normalized[:-5]
        add(normalized)
        add(base)
    else:
        add(f"{normalized}-orig")
        add(normalized)
        if "-" in normalized:
            base = normalized.split("-", 1)[0]
            add(f"{base}-orig")
            add(base)

    return candidates


def entry_language_preferences(
    entry: dict, preferred_language: Optional[str] = None
) -> List[str]:
    candidates: List[str] = []
    for value in (
        preferred_language,
        entry.get("language"),
        entry.get("original_language"),
        entry.get("default_language"),
    ):
        for candidate in normalize_language_value(value):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def base_language_key(language_key: str) -> str:
    normalized = language_key.lower()
    if normalized.endswith("-orig"):
        normalized = normalized[:-5]
    return normalized.split("-", 1)[0]


def track_is_translated(track_formats: Sequence[dict]) -> bool:
    urls = [fmt.get("url", "") for fmt in track_formats if fmt.get("url")]
    return bool(urls) and all("tlang=" in url for url in urls)


def track_marked_original(language_key: str, track_formats: Sequence[dict]) -> bool:
    if language_key.endswith("-orig"):
        return True

    for fmt in track_formats:
        searchable_fields = (
            str(fmt.get("name", "")),
            str(fmt.get("format_id", "")),
            str(fmt.get("format_note", "")),
        )
        if "original" in " ".join(searchable_fields).lower():
            return True

    return False


def caption_key_rank(language_key: str, preferred_keys: Sequence[str]) -> tuple:
    if language_key in preferred_keys:
        return (0, preferred_keys.index(language_key))

    language_base = base_language_key(language_key)
    for index, preferred_key in enumerate(preferred_keys):
        if language_base == base_language_key(preferred_key):
            return (1, index, 0 if language_key.endswith("-orig") else 1, language_key)

    return (2, 0 if language_key.endswith("-orig") else 1, language_key)


def select_caption_language_key(
    caption_tracks: dict, entry: dict, preferred_language: Optional[str] = None
) -> Optional[str]:
    if not caption_tracks:
        return None

    preferred_keys = entry_language_preferences(entry, preferred_language)

    for key in preferred_keys:
        formats = caption_tracks.get(key)
        if formats and not track_is_translated(formats):
            return key

    original_keys = [
        key
        for key, formats in caption_tracks.items()
        if key.endswith("-orig") and not track_is_translated(formats)
    ]
    if original_keys:
        return sorted(original_keys, key=lambda key: caption_key_rank(key, preferred_keys))[0]

    marked_original_keys = [
        key
        for key, formats in caption_tracks.items()
        if track_marked_original(key, formats) and not track_is_translated(formats)
    ]
    if marked_original_keys:
        return sorted(marked_original_keys, key=lambda key: caption_key_rank(key, preferred_keys))[0]

    non_translated_keys = [
        key for key, formats in caption_tracks.items() if not track_is_translated(formats)
    ]
    if len(non_translated_keys) == 1:
        return non_translated_keys[0]

    return None


def select_caption_track(
    entry: dict, preferred_language: Optional[str] = None
) -> Optional[CaptionTrackSelection]:
    for source_name in ("subtitles", "automatic_captions"):
        track_map = entry.get(source_name) or {}
        language_key = select_caption_language_key(track_map, entry, preferred_language)
        if language_key:
            return CaptionTrackSelection(source=source_name, language_key=language_key)
    return None


def entry_video_url(entry: dict, fallback_url: Optional[str] = None) -> Optional[str]:
    for candidate in (
        entry.get("webpage_url"),
        entry.get("original_url"),
        entry.get("url"),
    ):
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            return candidate

    if entry.get("id"):
        return f"https://www.youtube.com/watch?v={entry['id']}"

    return fallback_url


def fetch_video_info(video_url: str) -> Optional[dict]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(video_url, download=False)


def collect_video_entries(
    input_url: str, max_videos: Optional[int] = None
) -> Tuple[dict, List[dict]]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "extract_flat": True,
    }
    if max_videos is not None:
        ydl_opts["playlist_items"] = f"1-{max_videos}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(input_url, download=False)

    if not info_dict:
        return {}, []

    raw_entries = info_dict.get("entries")
    if not raw_entries:
        full_entry = fetch_video_info(entry_video_url(info_dict, input_url) or input_url)
        return full_entry or info_dict, [full_entry or info_dict]

    full_entries: List[dict] = []
    for raw_entry in raw_entries:
        if not raw_entry:
            continue
        video_url = entry_video_url(raw_entry)
        if not video_url:
            logger.warning("Skipping entry without a resolvable video URL: %s", raw_entry.get("id"))
            continue
        full_entry = fetch_video_info(video_url)
        if full_entry:
            full_entries.append(full_entry)

    return info_dict, full_entries


def download_audio_file(video_url: str, output_dir: Path, title: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = sanitize_filename(title)
    temp_stem = output_dir / f"{safe_stem}__download"

    for existing in output_dir.glob(f"{temp_stem.name}*.wav"):
        existing.unlink()

    ydl_opts = {
        "format": "140/251/139/bestaudio/bestaudio*",
        "outtmpl": str(temp_stem),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "progress_hooks": [download_progress_hook],
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    downloaded_files = sorted(output_dir.glob(f"{temp_stem.name}*.wav"))
    if not downloaded_files:
        raise RuntimeError(f"Audio download failed for {video_url}")

    final_audio_path = output_dir / f"{safe_stem}.wav"
    if final_audio_path.exists():
        final_audio_path.unlink()

    downloaded_files[0].rename(final_audio_path)
    logger.info("Audio downloaded: %s", final_audio_path)
    return final_audio_path


def download_audio(
    input_url: str, output_dir: Path, max_videos: Optional[int] = None
) -> Tuple[List[Path], List[str]]:
    """
    Download audio from a YouTube URL, playlist, or channel.

    Returns the list of downloaded audio files and the list of failed URLs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _, entries = collect_video_entries(input_url, max_videos=max_videos)
    audio_files: List[Path] = []
    failed_urls: List[str] = []

    for entry in entries:
        video_url = entry_video_url(entry, input_url)
        if not video_url:
            failed_urls.append(entry.get("id", "unknown"))
            continue
        try:
            audio_files.append(
                download_audio_file(video_url, output_dir, entry.get("title") or entry.get("id") or "audio")
            )
        except Exception as exc:
            logger.error("Download failed for %s: %s", video_url, exc)
            failed_urls.append(video_url)

    return audio_files, failed_urls


def transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    model_name: str,
    language: Optional[str] = None,
    draft_model: Optional[str] = None,
) -> Path:
    """
    Transcribe an audio file using mlx-qwen3-asr.
    """
    from mlx_qwen3_asr import transcribe
    from mlx_qwen3_asr.load_models import _ModelHolder

    _ModelHolder.set_cache_capacity(2 if draft_model else 1)

    base_filename = audio_path.stem
    transcript_path = output_dir / f"{base_filename}_transcription.txt"

    if transcript_path.exists():
        logger.info("Transcription already exists, skipping: %s", transcript_path)
        return transcript_path

    logger.info("Executing transcription with mlx-qwen3-asr, model: %s", model_name)

    kwargs = {"model": model_name}
    if draft_model:
        kwargs["draft_model"] = draft_model
        logger.info("Using speculative decoding with draft model: %s", draft_model)
    if language:
        kwargs["language"] = language
        logger.info("Using specified language: %s", language)
    else:
        logger.info("No language specified, automatic detection will be used")

    try:
        result = transcribe(str(audio_path), **kwargs)
        transcript_path.write_text(result.text, encoding="utf-8")
        if not language and getattr(result, "language", None):
            logger.info("Automatically detected language: %s", result.language)
        logger.info("Transcription completed: %s", transcript_path)
        return transcript_path
    except Exception as exc:
        logger.error("Error during transcription: %s", exc)
        raise RuntimeError(f"Transcription error: {exc}") from exc


def download_caption_track(
    video_url: str,
    output_dir: Path,
    base_stem: str,
    selection: CaptionTrackSelection,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    caption_stem = sanitize_filename(f"{base_stem}__captions")

    for existing in output_dir.glob(f"{caption_stem}*.vtt"):
        existing.unlink()

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": False,
        "subtitleslangs": [selection.language_key],
        "subtitlesformat": "vtt",
        "outtmpl": str(output_dir / caption_stem),
        "writesubtitles": selection.source == "subtitles",
        "writeautomaticsub": selection.source == "automatic_captions",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    caption_files = sorted(output_dir.glob(f"{caption_stem}*.vtt"))
    if not caption_files:
        raise RuntimeError(
            f"No caption file downloaded for {video_url} ({selection.source}:{selection.language_key})"
        )

    return caption_files[0]


def strip_stage_directions(text: str) -> str:
    stripped = STAGE_DIRECTION_RE.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_text_spacing(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;!?])", r"\1", compact)


def caption_segment_tail(previous_text: str, current_text: str) -> str:
    if not previous_text:
        return current_text
    if current_text == previous_text or previous_text.endswith(current_text):
        return ""
    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :].strip()

    previous_tokens = previous_text.split()
    current_tokens = current_text.split()
    max_overlap = min(len(previous_tokens), len(current_tokens))
    for overlap in range(max_overlap, 0, -1):
        if previous_tokens[-overlap:] == current_tokens[:overlap]:
            return " ".join(current_tokens[overlap:])

    return current_text


def normalize_caption_content(content: str, keep_markers: bool = False) -> str:
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    segments: List[str] = []

    for block in re.split(r"\n\s*\n", normalized_content):
        lines = [line.strip().lstrip("\ufeff") for line in block.splitlines()]
        if not lines:
            continue
        if lines[0].startswith("WEBVTT") or lines[0].startswith("NOTE"):
            continue

        cue_lines: List[str] = []
        for line in lines:
            if not line:
                continue
            if line in {"WEBVTT"} or line.startswith(("Kind:", "Language:", "NOTE")):
                continue
            if CUE_NUMBER_RE.match(line) or TIMESTAMP_RE.match(line):
                continue

            cleaned_line = html.unescape(VTT_TAG_RE.sub("", line))
            cleaned_line = normalize_text_spacing(cleaned_line)
            if not keep_markers:
                cleaned_line = VOICE_MARKER_RE.sub("", cleaned_line)
                cleaned_line = strip_stage_directions(cleaned_line)
            if cleaned_line:
                cue_lines.append(cleaned_line)

        if not cue_lines:
            continue

        cue_text = normalize_text_spacing(" ".join(cue_lines))
        if cue_text:
            segments.append(cue_text)

    merged_segments: List[str] = []
    previous_segment = ""
    for segment in segments:
        tail = caption_segment_tail(previous_segment, segment)
        if tail:
            merged_segments.append(tail)
        previous_segment = segment

    return normalize_text_spacing(" ".join(merged_segments))


def normalize_caption_file(caption_path: Path, keep_markers: bool = False) -> str:
    return normalize_caption_content(caption_path.read_text(encoding="utf-8"), keep_markers=keep_markers)


def write_caption_transcript(transcript_path: Path, transcript_text: str) -> Path:
    if not transcript_text.strip():
        raise RuntimeError(f"Transcript text is empty for {transcript_path}")

    transcript_path.write_text(transcript_text.strip() + "\n", encoding="utf-8")
    return transcript_path


def process_youtube_entry(
    entry: dict,
    output_dir: Path,
    model_name: str,
    language: Optional[str] = None,
    keep_audio: bool = False,
    draft_model: Optional[str] = None,
    transcript_source: str = "prefer-captions",
    markers: bool = False,
) -> Path:
    video_url = entry_video_url(entry)
    if not video_url:
        raise RuntimeError(f"Entry has no video URL: {entry.get('id', 'unknown')}")

    title = entry.get("title") or entry.get("id") or "video"
    safe_stem = sanitize_filename(title)
    transcript_path = output_dir / f"{safe_stem}_transcription.txt"

    if transcript_path.exists():
        logger.info("Transcription already exists, skipping: %s", transcript_path)
        write_output_copy(transcript_path, output_dir)
        return transcript_path

    if transcript_source != "asr-only":
        selection = select_caption_track(entry, preferred_language=language)
        if selection:
            logger.info(
                "Using %s track %s for %s",
                selection.source,
                selection.language_key,
                video_url,
            )
            caption_path = download_caption_track(video_url, output_dir, safe_stem, selection)
            try:
                transcript_text = normalize_caption_file(caption_path, keep_markers=markers)
            finally:
                if caption_path.exists():
                    caption_path.unlink()

            write_caption_transcript(transcript_path, transcript_text)
            write_output_copy(transcript_path, output_dir)
            logger.info("Caption transcript completed: %s", transcript_path)
            return transcript_path

        if transcript_source == "captions-only":
            raise RuntimeError(f"No usable caption track available for {video_url}")

    audio_path = download_audio_file(video_url, output_dir, title)
    try:
        asr_transcript_path = transcribe_audio(
            audio_path,
            output_dir,
            model_name,
            language=language,
            draft_model=draft_model,
        )
        write_output_copy(asr_transcript_path, output_dir)
        logger.info("ASR transcript completed: %s", asr_transcript_path)
        return asr_transcript_path
    finally:
        if not keep_audio and audio_path.exists():
            logger.info("Deleting audio file: %s", audio_path)
            audio_path.unlink()


def resolve_youtube_output_dir(
    output_dir: str, collection_info: dict, entry_count: int
) -> Path:
    base_output_dir = Path(output_dir)
    if entry_count > 1:
        collection_title = (
            collection_info.get("title")
            or collection_info.get("id")
            or collection_info.get("webpage_url_basename")
            or "playlist"
        )
        return base_output_dir / sanitize_filename(collection_title)
    return base_output_dir


def main(
    input_url: str,
    output_dir: str,
    model_name: str,
    language: Optional[str] = None,
    keep_audio: bool = False,
    max_videos: Optional[int] = None,
    draft_model: Optional[str] = None,
    transcript_source: str = "prefer-captions",
    markers: bool = False,
) -> List[Path]:
    """
    Transcribe a YouTube input, preferring original-language captions before ASR.
    """
    collection_info, entries = collect_video_entries(input_url, max_videos=max_videos)
    if not entries:
        raise RuntimeError(f"No video entries found for {input_url}")

    output_path = resolve_youtube_output_dir(output_dir, collection_info, len(entries))
    output_path.mkdir(exist_ok=True, parents=True)

    transcript_paths: List[Path] = []
    failures: List[str] = []

    for entry in entries:
        try:
            transcript_paths.append(
                process_youtube_entry(
                    entry,
                    output_path,
                    model_name,
                    language=language,
                    keep_audio=keep_audio,
                    draft_model=draft_model,
                    transcript_source=transcript_source,
                    markers=markers,
                )
            )
        except Exception as exc:
            entry_label = entry.get("webpage_url") or entry.get("title") or entry.get("id") or input_url
            logger.error("Transcription failed for %s: %s", entry_label, exc)
            failures.append(f"{entry_label}: {exc}")

    if failures and (transcript_source == "captions-only" or not transcript_paths):
        raise RuntimeError(" ; ".join(failures))

    return transcript_paths


def ensure_local_input_supported(input_path: Path, transcript_source: str) -> None:
    if transcript_source == "captions-only":
        raise ValueError(
            f"{transcript_source} is only supported for YouTube inputs, not local audio: {input_path}"
        )


def maybe_copy_to_clipboard(transcript_paths: Sequence[Path]) -> None:
    if not transcript_paths:
        return

    try:
        print()
        if len(transcript_paths) > 1:
            prompt_message = (
                f"Do you want to copy the LAST transcription ({transcript_paths[-1].name}) "
                "to your clipboard? (y/N): "
            )
        else:
            prompt_message = "Do you want to copy the transcription to your clipboard? (y/N): "

        choice = input(prompt_message).strip().lower()
        if choice != "y":
            return

        with open(transcript_paths[-1], "r", encoding="utf-8") as handle:
            content = handle.read()
        subprocess.run(["pbcopy"], input=content.encode("utf-8"), check=False)
        logger.info("✅ Transcription successfully copied to clipboard!")
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        logger.error("❌ Failed to copy to clipboard: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Caption-first YouTube or local .wav transcription with mlx-qwen3-asr fallback"
    )
    parser.add_argument("input", help="YouTube URL or path to a local .wav audio file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-m", "--model", default=DEFAULT_ASR_MODEL, help="ASR model to use")
    parser.add_argument(
        "-l",
        "--lang",
        help="Preferred original-language caption key or ASR language override",
    )
    parser.add_argument(
        "-k",
        "--keep-audio",
        action="store_true",
        help="Keep downloaded audio files after ASR transcription",
    )
    parser.add_argument("-n", "--max-videos", type=int, help="Maximum number of videos to process")
    parser.add_argument(
        "-d",
        "--draft-model",
        help="Draft model for speculative decoding (e.g. Qwen/Qwen3-ASR-0.6B)",
    )
    parser.add_argument(
        "--transcript-source",
        choices=("prefer-captions", "captions-only", "asr-only"),
        default="prefer-captions",
        help="Prefer YouTube captions before ASR, require captions, or force ASR",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="Preserve human-readable caption markers like >> and [music] when captions are used",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_dir():
        ensure_local_input_supported(input_path, args.transcript_source)
        logger.info("Detected directory of .wav files: %s", input_path)
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        transcript_files = []
        for wav_file in sorted(input_path.glob("*.wav")):
            logger.info("Transcribing %s", wav_file.name)
            transcript_file = transcribe_audio(
                wav_file,
                output_path,
                args.model,
                args.lang,
                args.draft_model,
            )
            transcript_files.append(transcript_file)
            logger.info("Transcription file: %s", transcript_file)

        logger.info("\n%s", "=" * 40)
        logger.info("✨ DIRECTORY PROCESSING COMPLETED ✨")
        logger.info("%s", "=" * 40)
        logger.info("📁 Transcribed %s .wav files from %s", len(transcript_files), input_path)
        logger.info("%s", "=" * 40)
        maybe_copy_to_clipboard(transcript_files)
        sys.exit(0)

    if input_path.suffix.lower() == ".wav" and input_path.exists():
        ensure_local_input_supported(input_path, args.transcript_source)
        logger.info("Detected local .wav file: %s", input_path)
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        transcript_file = transcribe_audio(
            input_path,
            output_path,
            args.model,
            args.lang,
            args.draft_model,
        )

        logger.info("\n%s", "=" * 40)
        logger.info("✨ PROCESSING COMPLETED ✨")
        logger.info("%s", "=" * 40)
        logger.info("📄 Transcription : %s", transcript_file)
        logger.info("%s", "=" * 40)
        maybe_copy_to_clipboard([transcript_file])
        sys.exit(0)

    transcript_files = main(
        args.input,
        args.output,
        args.model,
        args.lang,
        args.keep_audio,
        args.max_videos,
        args.draft_model,
        transcript_source=args.transcript_source,
        markers=args.markers,
    )

    logger.info("\n%s", "=" * 40)
    logger.info("✨ PROCESSING COMPLETED ✨")
    logger.info("%s", "=" * 40)
    for transcript_file in transcript_files:
        logger.info("📄 Transcription : %s", transcript_file)
    logger.info("%s", "=" * 40)
    maybe_copy_to_clipboard(transcript_files)
