"""Audio-visual media captioning using multimodal models.

Ported from ltx-trainer (Lightricks). This module provides captioning
capabilities for videos with audio using:
- Qwen2.5-Omni: Local model supporting text, audio, image, and video inputs
- Gemini Flash: Cloud-based API for audio-visual captioning

Note: Qwen support requires PyTorch/transformers and is not MLX-native.
The Gemini captioner is pure Python API calls with no framework dependency.

Requirements:
- Gemini Flash: google-generativeai (uv pip install google-generativeai)
  Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

# Instruction for audio-visual captioning (default)
DEFAULT_CAPTION_INSTRUCTION = """\
Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[SPEECH]: <Word-for-word transcription of everything spoken.
           Listen carefully and transcribe the exact words. If no speech, write "None">
[SOUNDS]: <Description of music, ambient sounds, sound effects. If none, write "None">
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in all four sections. For [SPEECH], transcribe the actual words spoken, not a summary."""

# Instruction for video-only captioning (no audio processing)
VIDEO_ONLY_CAPTION_INSTRUCTION = """\
Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in both sections."""


class CaptionerType(StrEnum):
    """Enum for different types of media captioners."""

    GEMINI_FLASH = "gemini_flash"


def create_captioner(captioner_type: CaptionerType, **kwargs: object) -> MediaCaptioningModel:
    """Factory function to create a media captioner.

    Args:
        captioner_type: The type of captioner to create.
        **kwargs: Additional arguments to pass to the captioner constructor.

    Returns:
        An instance of a ``MediaCaptioningModel``.
    """
    match captioner_type:
        case CaptionerType.GEMINI_FLASH:
            return GeminiFlashCaptioner(**kwargs)
        case _:
            raise ValueError(f"Unsupported captioner type: {captioner_type}")


class MediaCaptioningModel(ABC):
    """Abstract base class for audio-visual media captioning models."""

    @abstractmethod
    def caption(self, path: str | Path, **kwargs: object) -> str:
        """Generate a caption for the given video or image.

        Args:
            path: Path to the video/image file to caption.

        Returns:
            A string containing the generated caption.
        """

    @property
    @abstractmethod
    def supports_audio(self) -> bool:
        """Whether this captioner supports audio input."""

    @staticmethod
    def _is_image_file(path: str | Path) -> bool:
        """Check if the file is an image based on extension."""
        return str(path).lower().endswith((".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"))

    @staticmethod
    def _is_video_file(path: str | Path) -> bool:
        """Check if the file is a video based on extension."""
        return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    @staticmethod
    def _clean_raw_caption(caption: str) -> str:
        """Clean up the raw caption by removing common VLM patterns."""
        start = ["The", "This"]
        kind = ["video", "image", "scene", "animated sequence", "clip", "footage"]
        act = ["displays", "shows", "features", "depicts", "presents", "showcases", "captures", "contains"]

        for x, y, z in itertools.product(start, kind, act):
            caption = caption.replace(f"{x} {y} {z} ", "", 1)

        return caption


class GeminiFlashCaptioner(MediaCaptioningModel):
    """Audio-visual captioning using Google's Gemini Flash API.

    Gemini Flash is a cloud-based multimodal model that natively supports
    audio and video understanding. Requires a Google API key.

    Note: This captioner requires the ``google-generativeai`` package and a
    valid API key. Set the ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``
    environment variable, or pass the key directly.
    """

    MODEL_ID = "gemini-flash-lite-latest"

    def __init__(
        self,
        api_key: str | None = None,
        instruction: str | None = None,
    ):
        """Initialize the Gemini Flash captioner.

        Args:
            api_key: Google API key. If not provided, will look for
                ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` environment variable.
            instruction: Custom instruction prompt. If ``None``, uses the
                default instruction.
        """
        self.instruction = instruction
        self._init_client(api_key)

    @property
    def supports_audio(self) -> bool:
        return True

    def caption(
        self,
        path: str | Path,
        fps: int = 3,
        include_audio: bool = True,
        clean_caption: bool = True,
    ) -> str:
        """Generate a caption for the given video or image.

        Args:
            path: Path to the video/image file to caption.
            fps: Frames per second (not used for Gemini, kept for API compat).
            include_audio: Whether to include audio content in the caption.
            clean_caption: Whether to clean up the raw caption.

        Returns:
            A string containing the generated caption.
        """
        import time

        path = Path(path)
        is_video = self._is_video_file(path)
        use_audio = include_audio and is_video

        if self.instruction is not None:
            instruction = self.instruction
        else:
            instruction = DEFAULT_CAPTION_INSTRUCTION if use_audio else VIDEO_ONLY_CAPTION_INSTRUCTION

        # Upload the file to Gemini
        uploaded_file = self._genai.upload_file(path)

        # Wait for processing to complete
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = self._genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {uploaded_file.state.name}")

        # Generate caption
        response = self._model.generate_content([uploaded_file, instruction])

        caption_raw = response.text

        # Clean up the uploaded file
        self._genai.delete_file(uploaded_file.name)

        # Clean up caption if requested
        return self._clean_raw_caption(caption_raw) if clean_caption else caption_raw

    def _init_client(self, api_key: str | None) -> None:
        """Initialize the Gemini API client."""
        import os

        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "The `google-generativeai` package is required for Gemini Flash captioning. "
                "Install it with: `uv pip install google-generativeai`"
            ) from e

        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "Gemini API key is required. Provide it via the `api_key` argument "
                "or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=resolved_api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(self.MODEL_ID)


def example() -> None:
    """Example usage of the captioning module."""
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <video_path> [captioner_type]")
        print("  captioner_type: gemini_flash (default)")
        sys.exit(1)

    video_path = sys.argv[1]
    captioner_type = CaptionerType(sys.argv[2]) if len(sys.argv) > 2 else CaptionerType.GEMINI_FLASH

    print(f"Using {captioner_type.value} captioner:")
    captioner = create_captioner(captioner_type)
    caption = captioner.caption(video_path)
    print(f"CAPTION: {caption}")


if __name__ == "__main__":
    example()
