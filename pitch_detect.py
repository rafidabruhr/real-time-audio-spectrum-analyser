"""
Lightweight pitch (fundamental frequency) detection for the live overlay.

Uses time-domain autocorrelation (ACF) with parabolic interpolation — no
extra dependencies beyond numpy, so it works with whatever is already
installed for the FFT spectrum view.

Algorithm
---------
1. Remove DC offset from the frame.
2. Autocorrelate the frame with itself (via FFT for speed).
3. Restrict the search to the lag range implied by [fmin, fmax].
4. Pick the strongest peak in that range.
5. Parabolic interpolation around the peak for sub-sample lag precision.
6. Confidence = normalized autocorrelation value at that peak (0..1).
   Frames below the confidence/RMS thresholds report "no pitch" so the
   overlay does not flicker during silence or noise.
"""

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
    """
    Convert a frequency to the nearest note name and cents deviation.

    Returns
    -------
    (note_name, cents_off) : tuple[str, float]
        e.g. ("A4", -3.2). cents_off is in roughly [-50, 50].
    """
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
    """
    Estimate the fundamental frequency of one frame via autocorrelation.

    Parameters
    ----------
    samples : np.ndarray
        Time-domain frame (the same frame used for the FFT view is fine).
    sample_rate : int
        Samples per second.
    fmin, fmax : float
        Search range in Hz (default: human voice / typical instrument range).

    Returns
    -------
    (frequency_hz, confidence) : tuple[float | None, float]
        ``frequency_hz`` is None if the signal is too quiet or no clear
        periodicity was found in range.
    """
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
    """
    Smooths per-frame pitch estimates for a stable overlay: a median filter
    over the last few confident estimates, so one noisy frame doesn't make
    the on-screen note jump or flicker.
    """

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
