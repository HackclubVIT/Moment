"""Tests for recorder.py — audio capture."""

from __future__ import annotations

import numpy as np
import pytest

from ai.whisper.recorder import AudioRecorder


class TestMixStreams:
    """Unit tests for AudioRecorder._mix_streams (static method)."""

    def test_both_empty(self):
        mic = np.array([], dtype="float32")
        sys = np.array([], dtype="float32")
        result = AudioRecorder._mix_streams(mic, sys)
        assert result.size == 0

    def test_mic_only(self):
        mic = np.ones(100, dtype="float32") * 0.5
        sys = np.array([], dtype="float32")
        result = AudioRecorder._mix_streams(mic, sys)
        np.testing.assert_array_equal(result, mic)

    def test_sys_only(self):
        mic = np.array([], dtype="float32")
        sys = np.ones(100, dtype="float32") * 0.5
        result = AudioRecorder._mix_streams(mic, sys)
        np.testing.assert_array_equal(result, sys)

    def test_equal_length(self):
        mic = np.ones(100, dtype="float32") * 0.5
        sys = np.ones(100, dtype="float32") * 0.3
        result = AudioRecorder._mix_streams(mic, sys, mic_weight=0.7, sys_weight=0.3)
        expected = 0.5 * 0.7 + 0.3 * 0.3  # = 0.44
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_different_lengths_pads(self):
        mic = np.ones(50, dtype="float32") * 0.5
        sys = np.ones(100, dtype="float32") * 0.3
        result = AudioRecorder._mix_streams(mic, sys)
        assert len(result) == 100

    def test_peak_normalization(self):
        """Mixed signal exceeding 1.0 should be normalized."""
        mic = np.ones(100, dtype="float32") * 1.0
        sys = np.ones(100, dtype="float32") * 1.0
        result = AudioRecorder._mix_streams(mic, sys, mic_weight=1.0, sys_weight=1.0)
        assert np.abs(result).max() <= 1.0


class TestListDevices:
    """Test device listing (non-destructive)."""

    def test_returns_list(self):
        devices = AudioRecorder.list_devices()
        assert isinstance(devices, list)
        if devices:
            assert "name" in devices[0]
            assert "index" in devices[0]


class TestRecordingLifecycle:
    """Test start/stop state management."""

    def test_not_recording_initially(self):
        rec = AudioRecorder()
        assert not rec.is_recording
        assert rec.elapsed == 0.0

    def test_stop_without_start_raises(self):
        rec = AudioRecorder()
        with pytest.raises(RuntimeError, match="No recording"):
            rec.stop()

    def test_double_start_raises(self):
        rec = AudioRecorder()
        try:
            rec.start(system_device=None)
            with pytest.raises(RuntimeError, match="already in progress"):
                rec.start(system_device=None)
        finally:
            if rec.is_recording:
                rec.stop()
