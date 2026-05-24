"""Utility functions for creating and handling multimodal data (images, videos, audio).

This module provides functions to:
  - Create dummy media files with deterministic properties
  - Convert media files to data URIs for API requests
"""

import base64
import mimetypes
import os
import wave

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Data URI Conversion
# -----------------------------------------------------------------------------


def _file_to_data_uri(path: str) -> str:
    """Read a binary file and return a data-URI, e.g. data:image/png;base64,iVBORw0KGgo..."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def get_file_data_uris(filenames: list[str], root: str = "images") -> list[str]:
    """Return list of data-URIs for media files (images, videos, audio)."""
    return [_file_to_data_uri(os.path.join(root, name)) for name in filenames]


# -----------------------------------------------------------------------------
# Dummy Media File Creation
# -----------------------------------------------------------------------------


def create_dummy_image(
    height: int,
    width: int,
    id: int = 0,
) -> str:
    """Create a dummy image with a fixed color based on the id.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        id: ID for deterministic color generation.

    Returns:
        Filename of the created image (relative to mm_data/images/ directory).

    Raises:
        ValueError: If height or width is not positive.
        RuntimeError: If image cannot be saved.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    if height > width:
        height, width = width, height

    filename = f"{height}x{width}_{id}.png"
    os.makedirs("mm_data/images", exist_ok=True)
    path = os.path.join("mm_data/images", filename)
    if os.path.exists(path):
        return filename

    # Deterministic BGR color from id
    b = 32 + (id * 73) % 192
    g = 32 + (id * 151) % 192
    r = 32 + (id * 191) % 192
    img = np.full((height, width, 3), (b, g, r), dtype=np.uint8)

    # Fixed PNG compression for consistent output; 9 is maximum compression
    if not cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise RuntimeError(f"Failed to save image: {path}")

    return filename


def create_dummy_video(
    num_frames: int,
    width: int,
    height: int,
    id: int = 0,
) -> str:
    """Create a dummy video with fixed color frames based on the id.

    Args:
        num_frames: Number of frames in the video.
        width: Video width in pixels.
        height: Video height in pixels.
        id: ID for deterministic color generation.

    Returns:
        Filename of the created video (relative to videos/ directory).

    Raises:
        ValueError: If height, width, or num_frames is not positive.
        RuntimeError: If video cannot be created.
    """
    if height <= 0 or width <= 0 or num_frames <= 0:
        raise ValueError("height, width, and num_frames must be positive")
    if num_frames < 2:
        print("Warning: num_frames < 2 is not a valid video; using 2")
        num_frames = 2  # Minimum for a video

    if height > width:
        height, width = width, height

    filename = f"{num_frames}x{height}x{width}_{id}.mp4"
    os.makedirs("mm_data/videos", exist_ok=True)
    path = os.path.join("mm_data/videos", filename)
    if os.path.exists(path):
        return filename

    # Deterministic BGR color from id
    b = 32 + (id * 73) % 192
    g = 32 + (id * 151) % 192
    r = 32 + (id * 191) % 192
    img = np.full((height, width, 3), (b, g, r), dtype=np.uint8)

    # Use MP4 codec; may not work on all systems
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path, fourcc, 25.0, (width, height))
    if not video.isOpened():
        raise RuntimeError(f"Failed to open video for writing: {path}")

    for _ in range(num_frames):
        video.write(img)
    video.release()

    return filename


def create_dummy_audio(duration_sec: int, id: int = 0) -> str:
    """Create a dummy audio file with random noise based on the id.

    Args:
        duration_sec: Duration of the audio in seconds.
        id: ID for deterministic random seed.

    Returns:
        Filename of the created audio (relative to audios/ directory).

    Raises:
        ValueError: If duration_sec is not positive.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    filename = f"{duration_sec}s_{id}.wav"
    os.makedirs("mm_data/audios", exist_ok=True)
    path = os.path.join("mm_data/audios", filename)
    if os.path.exists(path):
        return filename

    sample_rate = 16000
    num_samples = int(duration_sec * sample_rate)
    rng = np.random.default_rng(id)
    data = rng.integers(-32768, 32767, num_samples, dtype=np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

    return filename
