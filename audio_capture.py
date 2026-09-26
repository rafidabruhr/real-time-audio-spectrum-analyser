import platform
import sys
import numpy as np

from config import CHANNELS, CHUNK_SIZE, SAMPLE_RATE

_sd = None
_pyaudio = None

try:
    import sounddevice as _sd
except ImportError:
    pass

if _sd is None:
    try:
        import pyaudio as _pyaudio
    except ImportError:
        pass


class AudioCaptureError(Exception):
    pass


class AudioStreamReadError(AudioCaptureError):
    pass


class LoopbackUnavailableError(AudioCaptureError):
    pass


def list_devices():
    if _sd is None:
        raise AudioCaptureError("sounddevice is not installed; cannot list devices.")
    out = []
    for i, dev in enumerate(_sd.query_devices()):
        out.append({
            "index": i,
            "name": dev.get("name", "?"),
            "hostapi": _sd.query_hostapis(dev.get("hostapi", 0))["name"],
            "max_input_channels": dev.get("max_input_channels", 0),
            "max_output_channels": dev.get("max_output_channels", 0),
            "default_samplerate": dev.get("default_samplerate", SAMPLE_RATE),
        })
    return out


def list_loopback_devices():
    devices = list_devices()
    system = platform.system()

    if system == "Windows":
        return [d for d in devices if d["hostapi"] == "Windows WASAPI" and d["max_output_channels"] > 0]

    keywords = ("monitor of", "blackhole", "soundflower", "loopback", "stereo mix")
    return [d for d in devices if any(k in d["name"].lower() for k in keywords)]


def _default_input_index_pyaudio(pa):
    try:
        return int(pa.get_default_input_device_info()["index"])
    except OSError:
        return None


def _has_input_device():
    try:
        devices = _sd.query_devices()
    except Exception as exc:
        raise AudioCaptureError(f"(OS settings / privacy). Details: {exc}") from exc
    return any(d.get("max_input_channels", 0) > 0 for d in devices)


class AudioStream:
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, channels=CHANNELS,
                 dtype="float32", source="mic", device=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        self.source = source
        self.device = device
        self._backend = None

        self._sd_stream = None
        self._pa = None
        self._pa_stream = None
        self._overflow_warned = False

        if source == "loopback":
            self._open_loopback()
        else:
            self._open_mic()

    @property
    def backend(self):
        if self._backend is None:
            raise RuntimeError("Stream is not open.")
        return self._backend

    def _open_mic(self):
        if _sd is not None:
            if self.device is None and not _has_input_device():
                raise AudioCaptureError(
                    "No audio input device found.\n"
                    "• Plug in a microphone or headset.\n"
                    "• On macOS: System Settings → Privacy → Microphone.\n"
                    "• On Windows: Settings → Privacy → Microphone.\n"
                    "• On Linux: check PulseAudio/PipeWire and input source."
                )
            try:
                self._sd_stream = _sd.InputStream(
                    samplerate=self.sample_rate, channels=self.channels,
                    dtype=self.dtype, blocksize=self.chunk_size, device=self.device,
                )
                self._sd_stream.start()
                self._backend = "sounddevice"
                return
            except Exception as exc:
                msg = str(exc).lower()
                if "device" in msg or "invalid" in msg or "no" in msg:
                    raise AudioCaptureError(
                        f"Could not open the microphone.\nCheck permissions and that a "
                        f"default input device is selected. Details: {exc}"
                    ) from exc
                raise AudioCaptureError(f"Failed to start audio capture: {exc}") from exc

        if _pyaudio is not None:
            pa = _pyaudio.PyAudio()
            device_index = self.device if self.device is not None else _default_input_index_pyaudio(pa)
            if device_index is None:
                pa.terminate()
                raise AudioCaptureError("No audio input device found (PyAudio).\nCheck microphone connection and OS permissions.")
            fmt = _pyaudio.paInt16 if self.dtype == "int16" else _pyaudio.paFloat32
            try:
                stream = pa.open(format=fmt, channels=self.channels, rate=self.sample_rate,
                                  input=True, input_device_index=device_index,
                                  frames_per_buffer=self.chunk_size)
            except OSError as exc:
                pa.terminate()
                raise AudioCaptureError(f"Could not open the microphone (PyAudio).\nDetails: {exc}") from exc
            self._pa = pa
            self._pa_stream = stream
            self._backend = "pyaudio"
            return

        raise AudioCaptureError(
            "Neither sounddevice nor PyAudio is installed.\n"
            "Install dependencies: pip install -r requirements.txt\n"
            "If sounddevice fails on your system, try: pip install PyAudio"
        )

    def _open_loopback(self):
        if _sd is None:
            raise LoopbackUnavailableError(
                "Loopback capture requires sounddevice (PyAudio has no loopback support). "
                "Install it: pip install sounddevice"
            )

        system = platform.system()

        # Windows: WASAPI loopback on the default (or given) output device
        if system == "Windows":
            devices = _sd.query_devices()
            device_index = self.device
            if device_index is None:
                try:
                    device_index = _sd.default.device[1]
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
                self._sd_stream = _sd.InputStream(
                    samplerate=self.sample_rate, channels=open_channels, dtype=self.dtype,
                    blocksize=self.chunk_size, device=device_index,
                    extra_settings=_sd.WasapiSettings(loopback=True),
                )
                self._sd_stream.start()
                self._backend = "sounddevice"
                return
            except Exception as exc:
                raise LoopbackUnavailableError(
                    f"Could not open system audio in loopback mode (WASAPI).\n"
                    f"Device: {dev_info.get('name', device_index)}. Details: {exc}"
                ) from exc

        # Linux/macOS have no "this is loopback" flag — the monitor/virtual
        # device just looks like a regular mic, so we need it named explicitly
        if self.device is None:
            candidates = list_loopback_devices()
            hint = "\n".join(f"  [{d['index']}] {d['name']}" for d in candidates) or "  (none found)"
            platform_hint = (
                "On Linux: select the PulseAudio/PipeWire 'Monitor of <output>' device.\n"
                "On macOS: install a virtual device such as BlackHole "
                "(e.g. brew install blackhole-2ch) and route output to it "
                "(use a Multi-Output Device to still hear audio live)."
            )
            raise LoopbackUnavailableError(
                f"Loopback on this platform needs an explicit device.\n{platform_hint}\n"
                f"Candidates found via --list-devices:\n{hint}\nPass one with --device <index>."
            )

        try:
            self._sd_stream = _sd.InputStream(
                samplerate=self.sample_rate, channels=self.channels, dtype=self.dtype,
                blocksize=self.chunk_size, device=self.device,
            )
            self._sd_stream.start()
            self._backend = "sounddevice"
        except Exception as exc:
            raise LoopbackUnavailableError(f"Could not open device {self.device} for loopback capture.\nDetails: {exc}") from exc

    def _warn_overflow_once(self):
        if self._overflow_warned:
            return
        self._overflow_warned = True
        print(
            "Audio buffer overflow/underrun: capture could not keep up with real time.\n"
            "• Close other apps using the mic/output or heavy CPU loads.\n"
            "• On Linux, JACK/PulseAudio glitch: try replugging the device.",
            file=sys.stderr,
        )

    def read_chunk(self):
        if self._backend == "sounddevice":
            try:
                data, overflowed = self._sd_stream.read(self.chunk_size)
            except Exception as exc:
                raise AudioStreamReadError(
                    f"Audio input stopped — device may have been unplugged, disabled, "
                    f"or (for loopback) its output stopped.\nDetails: {exc}"
                ) from exc
            if overflowed:
                self._warn_overflow_once()
            arr = np.asarray(data, dtype=np.float32 if self.dtype == "float32" else np.int16)
            if arr.ndim == 2 and arr.shape[1] > 1 and self.channels == 1:
                arr = arr.mean(axis=1).astype(arr.dtype)
            return arr.reshape(-1) if self.channels == 1 else arr

        if self._backend == "pyaudio":
            try:
                raw = self._pa_stream.read(self.chunk_size, exception_on_overflow=True)
            except OSError as exc:
                msg = str(exc).lower()
                if "overflow" in msg or "under" in msg:
                    self._warn_overflow_once()
                    raw = self._pa_stream.read(self.chunk_size, exception_on_overflow=False)
                else:
                    raise AudioStreamReadError(
                        f"Microphone input stopped — device may have been unplugged or "
                        f"disabled (PyAudio).\nDetails: {exc}"
                    ) from exc
            except Exception as exc:
                raise AudioStreamReadError(f"Microphone read failed (PyAudio).\nDetails: {exc}") from exc

            arr = np.frombuffer(raw, dtype=np.int16 if self.dtype == "int16" else np.float32)
            return arr.copy() if self.channels == 1 else arr.reshape(-1, self.channels)

        raise AudioStreamReadError("Audio stream is not open. Restart the application.")

    def close(self):
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

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()