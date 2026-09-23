from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import PITCH_CONFIDENCE_MIN, PITCH_FMAX, PITCH_FMIN, PITCH_RMS_MIN


@dataclass
class PitchEstimate:
    frequency_hz: Optional[float]  # None when no confident pitch found
    confidence: float              # 0..1, normalized autocorrelation peak
    note_name: Optional[str]       # e.g. "A4", None if no pitch
    cents_off: float               # deviation from nearest semitone, -50..50


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_A4_HZ = 440.0
_A4_MIDI = 69


def hz_to_note(frequency_hz: float) -> Tuple[str, float]:
    midi = _A4_MIDI + 12.0 * np.log2(frequency_hz / _A4_HZ)
    nearest_midi = int(round(midi))
    cents_off = (midi - nearest_midi) * 100.0
    name = _NOTE_NAMES[nearest_midi % 12]
    octave = nearest_midi // 12 - 1
    return f"{name}{octave}", cents_off


def autocorrelation_pitch(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float = PITCH_FMIN,
    fmax: float = PITCH_FMAX,
) -> Tuple[Optional[float], float]:
    x = np.asarray(samples, dtype=np.float64)
    x = x - np.mean(x)

    rms = float(np.sqrt(np.mean(x * x)))
    if rms < PITCH_RMS_MIN:
        return None, 0.0

    n = x.shape[0]
    # Autocorrelation via FFT (Wiener–Khinchin): fast even for 1024/2048 frames.
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=fft_size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:n]

    if acf[0] <= 0:
        return None, 0.0
    acf = acf / acf[0]  # normalize so lag-0 confidence is 1.0

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(n - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return None, 0.0

    window = acf[min_lag:max_lag]
    peak_idx = int(np.argmax(window))
    peak_lag = min_lag + peak_idx
    confidence = float(window[peak_idx])

    if confidence < PITCH_CONFIDENCE_MIN:
        return None, confidence

    # Parabolic interpolation using neighboring lags for sub-bin accuracy.
    if 0 < peak_lag < n - 1:
        y0, y1, y2 = acf[peak_lag - 1], acf[peak_lag], acf[peak_lag + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        shift = float(np.clip(shift, -1.0, 1.0))
    else:
        shift = 0.0

    refined_lag = peak_lag + shift
    frequency_hz = sample_rate / refined_lag
    return frequency_hz, confidence


class PitchTracker:

    def __init__(self, history: int = 5) -> None:
        self._history = history
        self._recent: List[float] = []

    def update(self, samples: np.ndarray, sample_rate: int) -> PitchEstimate:
        """Feed one frame, get back the smoothed current estimate."""
        freq, confidence = autocorrelation_pitch(samples, sample_rate)

        if freq is None:
            self._recent.clear()
            return PitchEstimate(None, confidence, None, 0.0)

        self._recent.append(freq)
        if len(self._recent) > self._history:
            self._recent.pop(0)

        smoothed = float(np.median(self._recent))
        note_name, cents_off = hz_to_note(smoothed)
        return PitchEstimate(smoothed, confidence, note_name, cents_off)
