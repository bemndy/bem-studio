"""Decode and encode audio with ffmpeg.

Uploaded audio (mp3, m4a, flac, ogg, ...) is decoded to raw float samples
for the stages, and output stems are encoded to formats a browser can
stream or download. ffmpeg is used because it handles virtually every
container and codec and is already required by the ML tooling.

Audio arrays are always float32 shaped (channels, samples), which is what
torch and Demucs expect. ffmpeg's raw streams are interleaved
([L0, R0, L1, R1, ...]); decode() and encode() convert between the two.
"""

import json
import logging
import shutil
import subprocess
import typing

from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# 44.1kHz stereo: CD quality, and what Demucs' pretrained models expect.
TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 2

# Lossy formats accept a bitrate; lossless ones ignore it.
LOSSY_SUFFIXES = {".opus", ".ogg", ".mp3", ".m4a", ".aac", ".webm"}


class FfmpegMissing(RuntimeError):
    """ffmpeg or ffprobe is not installed / not on PATH."""


class DecodeError(RuntimeError):
    """ffmpeg could not read the input file."""


class EncodeError(RuntimeError):
    """ffmpeg could not write the output file."""


def require_ffmpeg() -> None:
    """Raise FfmpegMissing if ffmpeg or ffprobe isn't on PATH.

    Gives a clear install hint instead of a bare FileNotFoundError from
    subprocess.
    """
    if shutil.which("ffmpeg") == None or shutil.which("ffprobe") == None:
        raise FfmpegMissing(
            "ffmpeg and/or ffprobe not found. Install it with "
            "'brew install ffmpeg' (macOS) or 'apt install ffmpeg' (Linux)."
        )


def probe(path: Path) -> dict:
    """Return ffprobe's JSON description of a file (format and streams).

    Used to reject unsuitable files (too long, no audio stream) before
    processing them.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )   

    return json.loads(result.stdout)


def decode(
    path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> typing.Tuple[np.ndarray, int]:
    """Decode any audio file to float32, shaped (channels, samples).

    Resamples to sample_rate and up/downmixes to channels. Returns
    (audio, sample_rate).

    Raises DecodeError with ffmpeg's message when the file is unreadable,
    missing, or has no audio stream.
    """
    require_ffmpeg()
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-",  # write to stdout
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # stderr is bytes; errors="replace" guards against undecodable filenames
        raise DecodeError(
            f"ffmpeg failed to decode {path}: {e.stderr.decode(errors='replace')}"
        ) from e

    # interleaved (samples, channels) -> (channels, samples); frombuffer
    # returns a read-only view, so copy to make it writable
    result = np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, channels).T.copy()
    if result.size == 0:
        raise DecodeError(f"ffmpeg decoded {path} to an empty array (no audio stream?)")
    
    return (result, sample_rate)
    
    


def encode(
    audio: np.ndarray,
    dest: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    bitrate: str = "128k",
) -> Path:
    """Write a (channels, samples) float array to dest.

    Container and codec are inferred from dest's extension:
      .opus / .mp3   lossy, for streaming to the browser
      .wav / .flac   lossless, for downloads

    Creates parent directories as needed and returns dest. bitrate applies
    only to lossy formats.

    Note: ffmpeg writes .wav as 16-bit PCM by default, so float32 input is
    quantized. Add -acodec pcm_s24le for 24-bit output.
    """
    require_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if audio.ndim != 2:
        raise ValueError(f"audio array must be 2-D (channels, samples), got {audio.ndim}-D")

    # back to interleaved order for ffmpeg's stdin
    payload = audio.T.astype(np.float32).tobytes()

    # -f/-ar/-ac come before -i because they describe the raw input, which
    # has no header for ffmpeg to read them from
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(audio.shape[0]),
        "-i",
        "-",
    ]
    if dest.suffix in LOSSY_SUFFIXES:
        cmd += ["-b:a", bitrate]
    cmd += [str(dest)]

    try:
        subprocess.run(cmd, input=payload, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise EncodeError(
            f"ffmpeg failed to encode {dest}: {e.stderr.decode(errors='replace')}"
        ) from e

    return dest



def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    """Length of a decoded array in seconds, rounded to milliseconds."""
    return round(audio.shape[1] / sample_rate, 3)