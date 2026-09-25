"""Tests for io_audio.py. Requires ffmpeg and ffprobe on PATH."""

import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO)

import numpy as np
import pytest

import io_audio as io


SAMPLE_RATE = 44100


@pytest.fixture
def tone():
    """3 seconds of stereo: 220Hz in the left channel, 330Hz in the right."""
    t = np.linspace(0, 3.0, SAMPLE_RATE * 3, endpoint=False)
    left = np.sin(2 * np.pi * 220 * t)
    right = np.sin(2 * np.pi * 330 * t)
    return np.stack([left, right]).astype(np.float32) * 0.5


class TestRequireFfmpeg:
    def test_passes_when_ffmpeg_is_installed(self):
        assert io.require_ffmpeg() is None

    def test_raises_when_a_binary_is_missing(self, monkeypatch):
        monkeypatch.setattr(io.shutil, "which", lambda name: None)
        with pytest.raises(io.FfmpegMissing) as caught:
            io.require_ffmpeg()
        # the message includes an install hint
        assert "ffmpeg" in str(caught.value).lower()


class TestEncode:
    def test_writes_a_file_and_returns_its_path(self, tone, tmp_path):
        dest = tmp_path / "out.wav"
        result = io.encode(tone, dest, SAMPLE_RATE)
        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size > 0

    def test_creates_missing_parent_directories(self, tone, tmp_path):
        dest = tmp_path / "stems" / "nested" / "drums.wav"
        io.encode(tone, dest, SAMPLE_RATE)
        assert dest.exists()

    def test_rejects_a_one_dimensional_array(self, tmp_path):
        flat = np.zeros(1000, dtype=np.float32)
        with pytest.raises(ValueError):
            io.encode(flat, tmp_path / "x.wav", SAMPLE_RATE)

    def test_lossy_output_is_much_smaller_than_lossless(self, tone, tmp_path):
        wav = io.encode(tone, tmp_path / "a.wav", SAMPLE_RATE)
        opus = io.encode(tone, tmp_path / "a.opus", SAMPLE_RATE)
        assert opus.stat().st_size < wav.stat().st_size / 4

    def test_handles_mono(self, tmp_path):
        t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
        mono = np.sin(2 * np.pi * 440 * t).astype(np.float32).reshape(1, -1)
        dest = io.encode(mono, tmp_path / "mono.wav", SAMPLE_RATE)
        audio, _ = io.decode(dest, channels=1)
        assert audio.shape[0] == 1

    def test_ffmpeg_failure_raises_encode_error(self, tone, tmp_path):
        # ffmpeg can't infer a format from an unknown extension
        with pytest.raises(io.EncodeError) as caught:
            io.encode(tone, tmp_path / "out.notaformat", SAMPLE_RATE)
        assert len(str(caught.value)) > 40


class TestDecode:
    def test_round_trip_through_flac_is_lossless(self, tone, tmp_path):
        path = io.encode(tone, tmp_path / "t.flac", SAMPLE_RATE)
        audio, rate = io.decode(path)

        assert rate == SAMPLE_RATE
        assert audio.shape == tone.shape
        assert audio.dtype == np.float32
        # FLAC is lossless, so this should come back essentially exact
        assert np.abs(audio - tone).mean() < 1e-6

    def test_shape_is_channels_first(self, tone, tmp_path):
        path = io.encode(tone, tmp_path / "t.flac", SAMPLE_RATE)
        audio, rate = io.decode(path)
        # 2 channels, 132300 samples
        assert audio.shape[0] == 2
        assert audio.shape[1] == SAMPLE_RATE * 3

    def test_channels_are_not_swapped(self, tone, tmp_path):
        """The 220Hz channel must still be channel 0 after a round trip.

        Catches a wrong reshape/transpose, which can give the right shape
        with scrambled data.
        """
        path = io.encode(tone, tmp_path / "t.flac", SAMPLE_RATE)
        audio, rate = io.decode(path)

        def dominant_hz(channel):
            spectrum = np.abs(np.fft.rfft(channel))
            return np.fft.rfftfreq(len(channel), 1 / rate)[np.argmax(spectrum)]

        assert dominant_hz(audio[0]) == pytest.approx(220, abs=2)
        assert dominant_hz(audio[1]) == pytest.approx(330, abs=2)

    def test_returned_array_is_writable(self, tone, tmp_path):
        """Stages need to be able to modify the array they're given."""
        path = io.encode(tone, tmp_path / "t.wav", SAMPLE_RATE)
        audio, _ = io.decode(path)
        audio[0, 0] = 0.123  # must not raise
        assert audio[0, 0] == pytest.approx(0.123)

    def test_resamples_and_downmixes_on_request(self, tone, tmp_path):
        path = io.encode(tone, tmp_path / "t.wav", SAMPLE_RATE)
        audio, rate = io.decode(path, sample_rate=22050, channels=1)
        assert rate == 22050
        assert audio.shape[0] == 1
        assert audio.shape[1] == pytest.approx(22050 * 3, rel=0.01)

    def test_lossy_round_trip_is_close_but_not_exact(self, tone, tmp_path):
        path = io.encode(tone, tmp_path / "t.mp3", SAMPLE_RATE)
        audio, _ = io.decode(path)
        error = np.abs(audio[:, : tone.shape[1]] - tone[:, : audio.shape[1]]).mean()
        assert error > 1e-6      # lossy: some loss
        assert error < 0.1       # ...but still recognizably the same audio

    def test_garbage_input_raises_decode_error(self, tmp_path):
        bad = tmp_path / "notaudio.mp3"
        bad.write_text("this is definitely not audio")
        with pytest.raises(io.DecodeError):
            io.decode(bad)

    def test_missing_file_raises_decode_error(self, tmp_path):
        with pytest.raises(io.DecodeError):
            io.decode(tmp_path / "does_not_exist.mp3")

    def test_decode_error_includes_ffmpeg_message(self, tmp_path):
        """DecodeError should carry ffmpeg's stderr."""
        with pytest.raises(io.DecodeError) as caught:
            io.decode(tmp_path / "does_not_exist.mp3")
        assert len(str(caught.value)) > 40


class TestProbe:
    def test_reports_codec_channels_and_duration(self, tone, tmp_path):
        path = io.encode(tone, tmp_path / "t.mp3", SAMPLE_RATE)
        info = io.probe(path)

        stream = info["streams"][0]
        assert stream["codec_name"] == "mp3"
        assert int(stream["channels"]) == 2
        assert float(info["format"]["duration"]) == pytest.approx(3.0, abs=0.1)


class TestDurationSeconds:
    def test_computes_length_from_the_sample_axis(self, tone):
        assert io.duration_seconds(tone, SAMPLE_RATE) == pytest.approx(3.0)

    def test_uses_samples_not_channels(self):
        # 2 channels, 44100 samples = 1.0s; dividing the channel axis
        # instead would give ~0.000045
        audio = np.zeros((2, SAMPLE_RATE), dtype=np.float32)
        assert io.duration_seconds(audio, SAMPLE_RATE) == pytest.approx(1.0)