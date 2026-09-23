"""
Microphone / system-loopback input stream handling for the spectrum analyzer.

Two capture sources are supported:
  * "mic"      — a normal input device (built-in mic, headset, USB mic).
  * "loopback" — "what you hear": the system's own audio output, captured
                 back in as an input signal. Lets the analyzer / pitch
                 tracker run on music or video playing on the computer
                 instead of (or in addition to) a microphone.

Loopback support is platform-dependent — there is no OS-agnostic "give me
system audio" API:

  * Windows — sounddevice/PortAudio can open any WASAPI *output* device in
    loopback mode via ``sd.WasapiSettings(loopback=True)``. No extra
    software needed.
  * Linux (PulseAudio/PipeWire) — the audio server exposes each output as a
    paired input named "Monitor of <device>". Select that device by index,
    same as a microphone; no special flag required.
  * macOS — CoreAudio has no built-in loopback. A virtual audio device such
    as BlackHole or Soundflower must be installed and selected as the input
    (route output to a Multi-Output Device so you can still hear it live).

``AudioStream(source="loopback")`` auto-detects the Windows case. On Linux
and macOS it raises a ``LoopbackUnavailableError`` listing candidate
"Monitor of ..." / virtual devices and asks the caller to pick one via
``device=<index>`` (``--device`` / ``--list-devices`` in main.py), since
PortAudio can't reliably guess which device *is* system audio on those
platforms.

Falls back to ``pyaudio`` (mic only — pyaudio has no loopback support) if
``sounddevice`` is not installed.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from config import CHANNELS, CHUNK_SIZE, SAMPLE_RATE

BackendName = Literal["sounddevice", "pyaudio"]
InputSource = Literal["mic", "loopback"]

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_sd = None
_pyaudio = None

try:
    import sounddevice as _sd  # type: ignore
except ImportError:
    _sd = None

if _sd is None:
    try:
        import pyaudio as _pyaudio  # type: ignore
    except ImportError:
        _pyaudio = None


class AudioCaptureError(Exception):
    """User-facing error for microphone / stream problems."""


class AudioStreamReadError(AudioCaptureError):
    """Failure while reading audio (disconnect, overflow, driver error)."""


class LoopbackUnavailableError(AudioCaptureError):
    """Loopback was requested but the platform/backend can't provide it."""


def list_devices() -> List[Dict[str, Any]]:
    """Return every audio device sounddevice can see (for ``--list-devices``)."""
    if _sd is None:
        raise AudioCaptureError("sounddevice is not installed; cannot list devices.")
    devices = _sd.query_devices()
    out: List[Dict[str, Any]] = []
    for i, dev in enumerate(devices):
        out.append(
            {
                "index": i,
                "name": dev.get("name", "?"),
                "hostapi": _sd.query_hostapis(dev.get("hostapi", 0))["name"],
                "max_input_channels": dev.get("max_input_channels", 0),
                "max_output_channels": dev.get("max_output_channels", 0),
                "default_samplerate": dev.get("default_samplerate", SAMPLE_RATE),
            }
        )
    return out


def list_loopback_devices() -> List[Dict[str, Any]]:
    """
    Return devices usable as a loopback ("what you hear") source.

    On Windows: every WASAPI output device (opened in loopback mode).
    On Linux/macOS: input devices whose name suggests they carry system
    audio (PulseAudio/PipeWire "Monitor of ...", or virtual devices like
    BlackHole / Soundflower / Stereo Mix).
    """
    all_devices = list_devices()
    system = platform.system()

    if system == "Windows":
        return [
            d for d in all_devices
            if d["hostapi"] == "Windows WASAPI" and d["max_output_channels"] > 0
        ]

    keywords = ("monitor of", "blackhole", "soundflower", "loopback", "stereo mix")
    return [d for d in all_devices if any(k in d["name"].lower() for k in keywords)]


def _default_input_index_pyaudio(pa: Any) -> Optional[int]:
    """Return default input device index, or None if unavailable."""
    try:
        info = pa.get_default_input_device_info()
        return int(info["index"])
    except OSError:
        return None


def _list_input_devices_sounddevice() -> bool:
    """Return True if at least one input-capable device exists."""
    assert _sd is not None
    try:
        devices = _sd.query_devices()
    except Exception as exc:
        raise AudioCaptureError(
            "Could not query audio devices. Check that your audio subsystem "
            "is running and that the application has microphone permission "
            f"(OS settings / privacy). Details: {exc}"
        ) from exc

    for dev in devices:
        if dev.get("max_input_channels", 0) > 0:
            return True
    return False


class AudioStream:
    """
    Blocking read interface to one audio input stream (mic or loopback).

    Parameters
    ----------
    sample_rate : int
        Samples per second (Hz).
    chunk_size : int
        Number of samples returned by each ``read_chunk`` call.
    channels : int
        Requested output channel count (1 = mono). For loopback the actual
        device stream may be opened in stereo and downmixed to mono in
        ``read_chunk`` when ``channels == 1``.
    dtype : str
        ``"float32"`` (normalized roughly -1..1) or ``"int16"``.
    source : "mic" | "loopback"
        Which signal to capture.
    device : int, optional
        Explicit device index (from ``list_devices`` / ``list_loopback_devices``).
        Required for loopback on Linux/macOS; optional on Windows (defaults
        to the system default output device) and for mic (defaults to the
        system default input device).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE,
        channels: int = CHANNELS,
        dtype: str = "float32",
        source: InputSource = "mic",
        device: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        self.source = source
        self.device = device
        self._backend: Optional[BackendName] = None

        self._sd_stream: Optional[object] = None
        self._pa: Optional[object] = None
        self._pa_stream: Optional[object] = None
        self._overflow_warned: bool = False

        self._open_stream()

    @property
    def backend(self) -> BackendName:
        """Which library is driving capture (``sounddevice`` or ``pyaudio``)."""
        if self._backend is None:
            raise RuntimeError("Stream is not open.")
        return self._backend

    def _open_stream(self) -> None:
        if self.source == "loopback":
            self._open_loopback()
        else:
            self._open_mic()

    def _open_mic(self) -> None:
        if _sd is not None:
            if self.device is None and not _list_input_devices_sounddevice():
                raise AudioCaptureError(
                    "No audio input device found.\n"
                    "• Plug in a microphone or headset.\n"
                    "• On macOS: System Settings → Privacy → Microphone.\n"
                    "• On Windows: Settings → Privacy → Microphone.\n"
                    "• On Linux: check PulseAudio/PipeWire and input source."
                )
            try:
                self._sd_stream = _sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    blocksize=self.chunk_size,
                    device=self.device,
                )
                self._sd_stream.start()
                self._backend = "sounddevice"
                return
            except Exception as exc:
                msg = str(exc).lower()
                if "device" in msg or "invalid" in msg or "no" in msg:
                    raise AudioCaptureError(
                        "Could not open the microphone.\n"
                        "Check permissions and that a default input device is "
                        f"selected. Details: {exc}"
                    ) from exc
                raise AudioCaptureError(f"Failed to start audio capture: {exc}") from exc

        if _pyaudio is not None:
            pa = _pyaudio.PyAudio()
            device_index = self.device if self.device is not None else _default_input_index_pyaudio(pa)
            if device_index is None:
                pa.terminate()
                raise AudioCaptureError(
                    "No audio input device found (PyAudio).\n"
                    "Check microphone connection and OS permissions."
                )
            fmt = _pyaudio.paInt16 if self.dtype == "int16" else _pyaudio.paFloat32
            try:
                stream = pa.open(
                    format=fmt,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                )
            except OSError as exc:
                pa.terminate()
                raise AudioCaptureError(
                    f"Could not open the microphone (PyAudio).\nDetails: {exc}"
                ) from exc
            self._pa = pa
            self._pa_stream = stream
            self._backend = "pyaudio"
            return

        raise AudioCaptureError(
            "Neither sounddevice nor PyAudio is installed.\n"
            "Install dependencies: pip install -r requirements.txt\n"
            "If sounddevice fails on your system, try: pip install PyAudio"
        )

    def _open_loopback(self) -> None:
        if _sd is None:
            raise LoopbackUnavailableError(
                "Loopback capture requires sounddevice (PyAudio has no "
                "loopback support). Install it: pip install sounddevice"
            )

        system = platform.system()

        if system == "Windows":
            devices = _sd.query_devices()
            device_index = self.device
            if device_index is None:
                try:
                    device_index = _sd.default.device[1]  # default output device
                except Exception:
                    device_index = None
            if device_index is None:
                raise LoopbackUnavailableError(
                    "Could not determine a default output device for loopback. "
                    "Pass an explicit device index — see --list-devices."
                )
            dev_info = devices[device_index]
            out_channels = max(1, int(dev_info.get("max_output_channels", 2)))
            open_channels = min(self.channels if self.channels > 1 else out_channels, out_channels)
            try:
                wasapi_settings = _sd.WasapiSettings(loopback=True)
                self._sd_stream = _sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=open_channels,
                    dtype=self.dtype,
                    blocksize=self.chunk_size,
                    device=device_index,
                    extra_settings=wasapi_settings,
                )
                self._sd_stream.start()
                self._backend = "sounddevice"
                return
            except Exception as exc:
                raise LoopbackUnavailableError(
                    "Could not open system audio in loopback mode (WASAPI).\n"
                    f"Device: {dev_info.get('name', device_index)}. Details: {exc}"
                ) from exc

        # Linux (PulseAudio/PipeWire "Monitor of ...") or macOS (BlackHole /
        # Soundflower): these appear as ordinary input devices, so they're
        # opened like a mic — but require the caller to name one explicitly,
        # since PortAudio has no "this is loopback" flag on these platforms.
        if self.device is None:
            candidates = list_loopback_devices()
            hint = (
                "\n".join(f"  [{d['index']}] {d['name']}" for d in candidates)
                if candidates
                else "  (none found)"
            )
            platform_hint = (
                "On Linux: select the PulseAudio/PipeWire 'Monitor of <output>' "
                "device.\nOn macOS: install a virtual device such as BlackHole "
                "(e.g. brew install blackhole-2ch) and route output to it "
                "(use a Multi-Output Device to still hear audio live)."
            )
            raise LoopbackUnavailableError(
                "Loopback on this platform needs an explicit device.\n"
                f"{platform_hint}\n"
                f"Candidates found via --list-devices:\n{hint}\n"
                "Pass one with --device <index>."
            )

        try:
            self._sd_stream = _sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                device=self.device,
            )
            self._sd_stream.start()
            self._backend = "sounddevice"
            return
        except Exception as exc:
            raise LoopbackUnavailableError(
                f"Could not open device {self.device} for loopback capture.\nDetails: {exc}"
            ) from exc

    def _warn_overflow_once(self) -> None:
        if self._overflow_warned:
            return
        self._overflow_warned = True
        print(
            "Audio buffer overflow/underrun: capture could not keep up with "
            "real time.\n"
            "• Close other apps using the mic/output or heavy CPU loads.\n"
            "• On Linux, JACK/PulseAudio glitch: try replugging the device.",
            file=sys.stderr,
        )

    def read_chunk(self) -> np.ndarray:
        """
        Read one buffer of audio.

        Returns
        -------
        np.ndarray
            Shape ``(chunk_size,)`` for mono. If the underlying stream is
            stereo (common for loopback) and ``self.channels == 1``,
            channels are averaged down to mono.

        Raises
        ------
        AudioStreamReadError
            Device disconnected, stream invalid, or unrecoverable I/O error.
        """
        if self._backend == "sounddevice":
            assert self._sd_stream is not None and _sd is not None
            try:
                data, overflowed = self._sd_stream.read(self.chunk_size)
            except Exception as exc:
                raise AudioStreamReadError(
                    "Audio input stopped — device may have been unplugged, "
                    "disabled, or (for loopback) its output stopped.\n"
                    f"Details: {exc}"
                ) from exc
            if overflowed:
                self._warn_overflow_once()
            arr = np.asarray(data, dtype=np.float32 if self.dtype == "float32" else np.int16)
            if arr.ndim == 2 and arr.shape[1] > 1 and self.channels == 1:
                arr = arr.mean(axis=1).astype(arr.dtype)
            if self.channels == 1:
                return arr.reshape(-1)
            return arr

        if self._backend == "pyaudio":
            assert self._pa_stream is not None and _pyaudio is not None
            try:
                raw = self._pa_stream.read(self.chunk_size, exception_on_overflow=True)
            except OSError as exc:
                msg = str(exc).lower()
                if "overflow" in msg or "under" in msg:
                    self._warn_overflow_once()
                    raw = self._pa_stream.read(self.chunk_size, exception_on_overflow=False)
                else:
                    raise AudioStreamReadError(
                        "Microphone input stopped — device may have been "
                        "unplugged or disabled (PyAudio).\n"
                        f"Details: {exc}"
                    ) from exc
            except Exception as exc:
                raise AudioStreamReadError(f"Microphone read failed (PyAudio).\nDetails: {exc}") from exc
            if self.dtype == "int16":
                arr = np.frombuffer(raw, dtype=np.int16)
            else:
                arr = np.frombuffer(raw, dtype=np.float32)
            if self.channels == 1:
                return arr.copy()
            return arr.reshape(-1, self.channels)

        raise AudioStreamReadError("Audio stream is not open. Restart the application.")

    def close(self) -> None:
        """Stop and release the input device."""
        if self._backend == "sounddevice" and self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

        if self._backend == "pyaudio":
            if self._pa_stream is not None:
                try:
                    self._pa_stream.stop_stream()
                    self._pa_stream.close()
                except Exception:
                    pass
                self._pa_stream = None
            if self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None

        self._backend = None

    def __enter__(self) -> "AudioStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
