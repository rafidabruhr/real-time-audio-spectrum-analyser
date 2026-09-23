from __future__ import annotations

import numpy as np

from config import WINDOW_TYPE

WindowType = str


def apply_window(samples: np.ndarray, window_type: str = WINDOW_TYPE) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    n = x.shape[0]
    wtype = window_type.lower()

    if wtype == "hann":
        # Hann (Hanning): goes to zero at both ends; good general-purpose choice.
        window = np.hanning(n)
    elif wtype == "hamming":
        # Hamming: similar to Hann but does not reach exactly zero at edges.
        window = np.hamming(n)
    elif wtype == "rectangular":
        # No taper — maximum leakage, but preserves amplitude of a bin-centered tone.
        window = np.ones(n, dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown window_type {window_type!r}. "
            'Use "hann", "hamming", or "rectangular".'
        )

    return x * window


def compute_fft_magnitude(windowed_samples: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(windowed_samples)
    # Magnitude = envelope strength at each frequency (phase discarded for display).
    return np.abs(spectrum)


def magnitude_to_db(magnitude: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mag = np.asarray(magnitude, dtype=np.float64)
    return 20.0 * np.log10(mag + epsilon)


def get_freq_bins(sample_rate: int, chunk_size: int) -> np.ndarray:
    return np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)


def analyze_frame(
    samples: np.ndarray,
    sample_rate: int,
    chunk_size: int,
    window_type: str = WINDOW_TYPE,
) -> tuple[np.ndarray, np.ndarray]:
    windowed = apply_window(samples, window_type)
    magnitude = compute_fft_magnitude(windowed)
    magnitude_db = magnitude_to_db(magnitude)
    freq_hz = get_freq_bins(sample_rate, chunk_size)
    return freq_hz, magnitude_db
