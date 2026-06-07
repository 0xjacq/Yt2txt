from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yt2txt


class CaptionSelectionTests(unittest.TestCase):
    def test_select_caption_track_prefers_manual_subtitles(self) -> None:
        entry = {
            "language": "en",
            "subtitles": {
                "en": [{"url": "https://example.com/manual-en.vtt"}],
            },
            "automatic_captions": {
                "en-orig": [{"url": "https://example.com/auto-en-orig.vtt"}],
            },
        }

        selection = yt2txt.select_caption_track(entry)

        self.assertEqual(
            selection,
            yt2txt.CaptionTrackSelection(source="subtitles", language_key="en"),
        )

    def test_select_caption_track_uses_auto_caption_when_manual_missing(self) -> None:
        entry = {
            "language": "en",
            "subtitles": {},
            "automatic_captions": {
                "en-orig": [{"url": "https://example.com/auto-en-orig.vtt"}],
            },
        }

        selection = yt2txt.select_caption_track(entry)

        self.assertEqual(
            selection,
            yt2txt.CaptionTrackSelection(source="automatic_captions", language_key="en-orig"),
        )

    def test_select_caption_track_ignores_translated_tracks(self) -> None:
        entry = {
            "language": "en",
            "subtitles": {},
            "automatic_captions": {
                "fr": [{"url": "https://example.com/subtitles.vtt?tlang=fr"}],
            },
        }

        selection = yt2txt.select_caption_track(entry)

        self.assertIsNone(selection)


class CaptionNormalizationTests(unittest.TestCase):
    SAMPLE_VTT = """WEBVTT

00:00:00.000 --> 00:00:01.000
>> Hello there

00:00:01.000 --> 00:00:02.000
>> Hello there world

00:00:02.000 --> 00:00:03.000
[music]
"""

    def test_normalize_caption_content_removes_vtt_artifacts_by_default(self) -> None:
        transcript = yt2txt.normalize_caption_content(self.SAMPLE_VTT)
        self.assertEqual(transcript, "Hello there world")

    def test_normalize_caption_content_preserves_markers_when_requested(self) -> None:
        transcript = yt2txt.normalize_caption_content(self.SAMPLE_VTT, keep_markers=True)
        self.assertEqual(transcript, ">> Hello there world [music]")


class ProcessingTests(unittest.TestCase):
    def test_process_youtube_entry_caption_writes_transcript_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            caption_path = output_dir / "captions.en-orig.vtt"
            caption_path.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n>> Hello there\n\n"
                "00:00:01.000 --> 00:00:02.000\n>> Hello there world\n",
                encoding="utf-8",
            )
            entry = {
                "title": "Caption Video",
                "webpage_url": "https://www.youtube.com/watch?v=caption",
            }

            with mock.patch.object(
                yt2txt,
                "select_caption_track",
                return_value=yt2txt.CaptionTrackSelection(
                    source="automatic_captions",
                    language_key="en-orig",
                ),
            ), mock.patch.object(
                yt2txt,
                "download_caption_track",
                return_value=caption_path,
            ), mock.patch.object(yt2txt, "transcribe_audio") as transcribe_mock:
                transcript_path = yt2txt.process_youtube_entry(
                    entry,
                    output_dir,
                    "Qwen/Qwen3-ASR-0.6B",
                )

            self.assertEqual(transcript_path.name, "Caption_Video_transcription.txt")
            self.assertEqual(
                transcript_path.read_text(encoding="utf-8"),
                "Hello there world\n",
            )
            self.assertEqual(
                (output_dir / "output.txt").read_text(encoding="utf-8"),
                "Hello there world\n",
            )
            self.assertFalse(caption_path.exists())
            transcribe_mock.assert_not_called()

    def test_process_youtube_entry_falls_back_to_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            audio_path = output_dir / "Fallback_Video.wav"
            transcript_path = output_dir / "Fallback_Video_transcription.txt"
            entry = {
                "title": "Fallback Video",
                "webpage_url": "https://www.youtube.com/watch?v=fallback",
            }

            def fake_download_audio_file(video_url: str, output_dir_arg: Path, title: str) -> Path:
                self.assertEqual(video_url, entry["webpage_url"])
                self.assertEqual(output_dir_arg, output_dir)
                self.assertEqual(title, entry["title"])
                audio_path.write_bytes(b"audio")
                return audio_path

            def fake_transcribe_audio(
                audio_path_arg: Path,
                output_dir_arg: Path,
                model_name: str,
                language: str | None = None,
                draft_model: str | None = None,
            ) -> Path:
                self.assertEqual(audio_path_arg, audio_path)
                self.assertEqual(output_dir_arg, output_dir)
                self.assertEqual(model_name, "Qwen/Qwen3-ASR-0.6B")
                self.assertIsNone(language)
                self.assertIsNone(draft_model)
                transcript_path.write_text("From ASR\n", encoding="utf-8")
                return transcript_path

            with mock.patch.object(yt2txt, "select_caption_track", return_value=None), mock.patch.object(
                yt2txt,
                "download_audio_file",
                side_effect=fake_download_audio_file,
            ), mock.patch.object(
                yt2txt,
                "transcribe_audio",
                side_effect=fake_transcribe_audio,
            ):
                returned_path = yt2txt.process_youtube_entry(
                    entry,
                    output_dir,
                    "Qwen/Qwen3-ASR-0.6B",
                )

            self.assertEqual(returned_path, transcript_path)
            self.assertEqual(
                (output_dir / "output.txt").read_text(encoding="utf-8"),
                "From ASR\n",
            )
            self.assertFalse(audio_path.exists())

    def test_process_youtube_entry_captions_only_requires_captions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            entry = {
                "title": "No Captions",
                "webpage_url": "https://www.youtube.com/watch?v=no-captions",
            }

            with mock.patch.object(yt2txt, "select_caption_track", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "No usable caption track available"):
                    yt2txt.process_youtube_entry(
                        entry,
                        output_dir,
                        "Qwen/Qwen3-ASR-0.6B",
                        transcript_source="captions-only",
                    )


class MainFlowTests(unittest.TestCase):
    def test_main_processes_each_playlist_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_output_dir = Path(tmp_dir)
            entries = [
                {"id": "one", "title": "One", "webpage_url": "https://www.youtube.com/watch?v=one"},
                {"id": "two", "title": "Two", "webpage_url": "https://www.youtube.com/watch?v=two"},
            ]
            processed_ids: list[str] = []

            def fake_process_youtube_entry(
                entry: dict,
                output_dir: Path,
                model_name: str,
                language: str | None = None,
                keep_audio: bool = False,
                draft_model: str | None = None,
                transcript_source: str = "prefer-captions",
                markers: bool = False,
            ) -> Path:
                self.assertEqual(model_name, "Qwen/Qwen3-ASR-0.6B")
                self.assertEqual(output_dir, base_output_dir / "Playlist_Title")
                processed_ids.append(entry["id"])
                transcript_path = output_dir / f"{entry['id']}_transcription.txt"
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_path.write_text(entry["id"], encoding="utf-8")
                return transcript_path

            with mock.patch.object(
                yt2txt,
                "collect_video_entries",
                return_value=({"title": "Playlist Title"}, entries),
            ), mock.patch.object(
                yt2txt,
                "process_youtube_entry",
                side_effect=fake_process_youtube_entry,
            ):
                paths = yt2txt.main(
                    "https://www.youtube.com/playlist?list=PL123",
                    str(base_output_dir),
                    "Qwen/Qwen3-ASR-0.6B",
                )

            self.assertEqual(processed_ids, ["one", "two"])
            self.assertEqual([path.name for path in paths], ["one_transcription.txt", "two_transcription.txt"])


class LocalInputTests(unittest.TestCase):
    def test_local_audio_rejects_captions_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "captions-only"):
            yt2txt.ensure_local_input_supported(Path("/tmp/example.wav"), "captions-only")

    def test_local_audio_allows_default_source_mode(self) -> None:
        yt2txt.ensure_local_input_supported(Path("/tmp/example.wav"), "prefer-captions")


if __name__ == "__main__":
    unittest.main()
