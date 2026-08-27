"""Audio capture module — mic + system audio recording.

Records from the default microphone and (optionally) a loopback / virtual
audio device for system sound. The two streams are mixed into a single
16 kHz mono WAV file suitable for WhisperX.

Cross-platform system audio notes:
  macOS  — Requires BlackHole (virtual audio driver). The user sets BlackHole
           as their system output; we capture from the BlackHole input device.
  Windows — WASAPI loopback is natively supported.
  Linux  — PulseAudio monitor sources work out of the box.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import WhisperConfig, load_config

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Cross-platform audio recorder for mic + optional system audio.

    Usage::

        rec = AudioRecorder()
        rec.start()           # starts recording in background
        # ... meeting happens ...
        path = rec.stop()     # stops and saves WAV
    """

    def __init__(self, config: WhisperConfig | None = None) -> None:
        self._cfg = config or load_config()
        self._sample_rate = self._cfg.sample_rate
        self._channels = self._cfg.channels

        self._mic_chunks: list[np.ndarray] = []
        self._sys_chunks: list[np.ndarray] = []

        self._mic_stream: sd.InputStream | None = None
        self._sys_stream: sd.InputStream | None = None

        self._is_recording = False
        self._lock = threading.Lock()
        self._start_time: float | None = None

    # ── device helpers ──────────────────────────────────────────────────

    @staticmethod
    def list_devices() -> list[dict]:
        """Return available audio devices with their indices and names."""
        devices = sd.query_devices()
        return [
            {
                "index": i,
                "name": d["name"],
                "max_input_channels": d["max_input_channels"],
                "max_output_channels": d["max_output_channels"],
                "default_samplerate": d["default_samplerate"],
                "hostapi": sd.query_hostapis(d["hostapi"])["name"],
            }
            for i, d in enumerate(devices)
        ]

    @staticmethod
    def find_loopback_device() -> int | None:
        """Attempt to auto-detect a loopback / virtual audio device.

        Returns the device index or None if nothing suitable is found.
        """
        os_name = platform.system()
        devices = sd.query_devices()

        for i, d in enumerate(devices):
            if d["max_input_channels"] < 1:
                continue
            name_lower = d["name"].lower()

            if os_name == "Darwin":
                # BlackHole or similar virtual drivers on macOS
                if any(kw in name_lower for kw in ("blackhole", "loopback", "soundflower")):
                    return i
            elif os_name == "Windows":
                hostapi = sd.query_hostapis(d["hostapi"])["name"].lower()
                if "wasapi" in hostapi and "loopback" in name_lower:
                    return i
            else:
                # Linux — PulseAudio monitor
                if "monitor" in name_lower:
                    return i

        return None

    # ── recording lifecycle ─────────────────────────────────────────────

    def start(
        self,
        mic_device: int | None = None,
        system_device: int | str | None = "auto",
    ) -> None:
        """Start recording from mic and (optionally) system audio.

        Args:
            mic_device: Specific mic device index, or None for system default.
            system_device: Device index for loopback, ``"auto"`` to auto-detect,
                           or ``None`` to skip system audio entirely.

        Raises:
            RuntimeError: If already recording.
            sd.PortAudioError: If the chosen device cannot be opened.
        """
        if self._is_recording:
            raise RuntimeError("Recording is already in progress")

        # Resolve system device
        sys_dev_idx: int | None = None
        if system_device == "auto":
            sys_dev_idx = self.find_loopback_device()
            if sys_dev_idx is not None:
                logger.info("Auto-detected loopback device: %s", sd.query_devices(sys_dev_idx)["name"])
            else:
                logger.warning("No loopback device found — recording mic only")
        elif isinstance(system_device, int):
            sys_dev_idx = system_device

        with self._lock:
            self._mic_chunks.clear()
            self._sys_chunks.clear()

            # Mic stream
            self._mic_stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                device=mic_device,
                dtype="float32",
                callback=self._mic_callback,
            )

            # System audio stream (if available)
            if sys_dev_idx is not None:
                try:
                    self._sys_stream = sd.InputStream(
                        samplerate=self._sample_rate,
                        channels=self._channels,
                        device=sys_dev_idx,
                        dtype="float32",
                        callback=self._sys_callback,
                    )
                except sd.PortAudioError as exc:
                    logger.warning("Could not open system audio device: %s", exc)
                    self._sys_stream = None

            self._mic_stream.start()
            if self._sys_stream:
                self._sys_stream.start()

            self._is_recording = True
            self._start_time = time.monotonic()

        logger.info("Recording started (mic=%s, system=%s)", mic_device or "default", sys_dev_idx)

    def stop(self, output_dir: str | Path | None = None) -> str:
        """Stop recording, mix streams, and save to a WAV file.

        Args:
            output_dir: Directory to write the file into.  Falls back to
                        the configured ``recordings_dir``.

        Returns:
            Absolute path to the saved WAV file.

        Raises:
            RuntimeError: If not currently recording.
        """
        if not self._is_recording:
            raise RuntimeError("No recording in progress")

        duration = time.monotonic() - (self._start_time or 0)

        with self._lock:
            # Tear down streams
            self._mic_stream.stop()
            self._mic_stream.close()
            if self._sys_stream:
                self._sys_stream.stop()
                self._sys_stream.close()

            mic_data = np.concatenate(self._mic_chunks) if self._mic_chunks else np.array([], dtype="float32")
            sys_data = np.concatenate(self._sys_chunks) if self._sys_chunks else np.array([], dtype="float32")

            self._mic_stream = None
            self._sys_stream = None
            self._is_recording = False

        mixed = self._mix_streams(mic_data, sys_data)
        out = self.save(mixed, output_dir)
        logger.info("Recording saved (%.1fs): %s", duration, out)
        return out

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def elapsed(self) -> float:
        """Seconds since recording started, or 0.0 if idle."""
        if self._start_time and self._is_recording:
            return time.monotonic() - self._start_time
        return 0.0

    # ── stream callbacks (called from audio thread) ─────────────────────

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("Mic stream status: %s", status)
        with self._lock:
            self._mic_chunks.append(indata.copy())

    def _sys_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("System stream status: %s", status)
        with self._lock:
            self._sys_chunks.append(indata.copy())

    # ── mixing ──────────────────────────────────────────────────────────

    @staticmethod
    def _mix_streams(
        mic: np.ndarray,
        system: np.ndarray,
        mic_weight: float = 0.7,
        sys_weight: float = 0.3,
    ) -> np.ndarray:
        """Weighted mix of mic and system audio.

        If either stream is empty, the other is returned as-is.
        Streams of different lengths are zero-padded to match.
        """
        # Ensure mono 1D arrays (sounddevice provides shape (frames, channels))
        if mic.ndim > 1:
            mic = mic.mean(axis=1)
        if system.ndim > 1:
            system = system.mean(axis=1)

        if mic.size == 0 and system.size == 0:
            return np.array([], dtype="float32")
        if mic.size == 0:
            return system.astype("float32")
        if system.size == 0:
            return mic.astype("float32")

        # Ensure same length
        target_len = max(len(mic), len(system))
        if len(mic) < target_len:
            mic = np.pad(mic, (0, target_len - len(mic)))
        if len(system) < target_len:
            system = np.pad(system, (0, target_len - len(system)))

        mixed = mic * mic_weight + system * sys_weight

        # Peak-normalize to avoid clipping
        peak = np.abs(mixed).max()
        if peak > 1.0:
            mixed /= peak

        return mixed.astype("float32")

    # ── save ────────────────────────────────────────────────────────────

    def save(self, audio: np.ndarray, output_dir: str | Path | None = None) -> str:
        """Write audio data to a uniquely-named WAV file.

        Returns:
            Absolute path to the saved file.
        """
        out_dir = Path(output_dir) if output_dir else self._cfg.recordings_path()
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"recording_{uuid.uuid4().hex[:8]}.wav"
        filepath = out_dir / filename

        sf.write(str(filepath), audio, self._sample_rate, subtype="PCM_16")
        return str(filepath.resolve())
