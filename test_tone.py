"""
Play a pure sine tone for calibrating the spectrum analyzer.

Use headphones or moderate volume: a 1 kHz tone should produce a peak on the
bar chart within one FFT bin of 1000 Hz when captured by the mic or loopback.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from config import SAMPLE_RATE

_sd = None
try:
    import sounddevice as _sd
except ImportError:
    _sd = None


def generate_sine_wave(
    frequency_hz: float,
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.4,
) -> np.ndarray:
    """
    Build a mono sine wave in float32.

    Parameters
    ----------
    frequency_hz : float
        Tone frequency in hertz.
    duration_sec : float
        Length of the signal in seconds.
    sample_rate : int
        Samples per second.
    amplitude : float
        Peak amplitude in roughly [-1, 1] before clipping.

    Returns
    -------
    np.ndarray
        float32 samples, shape ``(num_samples,)``.
    """
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    # Pure tone: single frequency component in the time domain.
    wave = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    return wave.astype(np.float32)


def play_sine_tone(
    frequency_hz: float = 1000.0,
    duration_sec: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.4,
) -> None:
    """
    Generate and play a sine tone through the default output device.

    Parameters
    ----------
    frequency_hz : float
        Tone frequency (default 1000 Hz).
    duration_sec : float
        Playback length; use a long value while you run the analyzer.
    sample_rate : int
        Must match analyzer ``SAMPLE_RATE`` for meaningful bin alignment.
    amplitude : float
        Output level; keep moderate to protect ears and avoid clipping.
    """
    if _sd is None:
        print(
            "sounddevice is required for playback. Install: pip install sounddevice",
            file=sys.stderr,
        )
        sys.exit(1)

    tone = generate_sine_wave(frequency_hz, duration_sec, sample_rate, amplitude)
    print(
        f"Playing {frequency_hz:.1f} Hz sine for {duration_sec:.1f} s "
        f"at {sample_rate} Hz (Ctrl+C to stop early)."
    )
    try:
        _sd.play(tone, sample_rate, blocking=True)
    except KeyboardInterrupt:
        _sd.stop()
        print("\nPlayback stopped.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a calibration sine tone through the speakers."
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=1000.0,
        help="Tone frequency in Hz (default: 1000).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Playback duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.4,
        help="Peak amplitude 0..1 (default: 0.4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    play_sine_tone(
        frequency_hz=args.freq,
        duration_sec=args.duration,
        amplitude=args.amplitude,
    )
