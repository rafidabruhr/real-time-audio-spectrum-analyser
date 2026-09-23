"""
Digital signal processing for short-time spectrum analysis.

Each function implements one step of the classic pipeline:
time-domain samples → window → FFT → magnitude → decibels → frequency axis.
"""

from __future__ import annotations

import numpy as np

from config import WINDOW_TYPE

WindowType = str


def apply_window(samples: np.ndarray, window_type: str = WINDOW_TYPE) -> np.ndarray:
    """
    Multiply the time-domain frame by a window function before the FFT.

    A finite chunk of audio is implicitly multiplied by a rectangular window
    (suddenly on at the edges). That discontinuity spreads energy across all
    FFT bins (spectral leakage). Tapering toward zero at the edges (Hann,
    Hamming) reduces leakage so a pure tone appears as a narrower peak.

    Parameters
    ----------
    samples : np.ndarray
        One frame of audio, shape ``(chunk_size,)``.
    window_type : str
        ``"hann"``, ``"hamming"``, or ``"rectangular"``.

    Returns
    -------
    np.ndarray
        Windowed samples, same shape as ``samples``, float64.
    """
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
    """
    Compute one-sided magnitude spectrum of a real-valued frame.

    Real audio has a symmetric spectrum; ``np.fft.rfft`` returns only
    non-negative frequency bins (0 Hz through Nyquist), which is half the
    work and the right data for plotting.

    Parameters
    ----------
    windowed_samples : np.ndarray
        Windowed time-domain frame.

    Returns
    -------
    np.ndarray
        Magnitude at each rFFT bin, length ``chunk_size // 2 + 1``.
    """
    spectrum = np.fft.rfft(windowed_samples)
    # Magnitude = envelope strength at each frequency (phase discarded for display).
    return np.abs(spectrum)


def magnitude_to_db(magnitude: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Convert linear magnitude to decibels (dB).

    Decibels compress huge dynamic range: dB = 20·log10(magnitude), using 20
    because we are treating FFT magnitude like an amplitude envelope. A small
    ``epsilon`` avoids log(0) when a bin is null after windowing.

    Parameters
    ----------
    magnitude : np.ndarray
        Linear FFT magnitudes.
    epsilon : float
        Floor added inside the logarithm.

    Returns
    -------
    np.ndarray
        Values in dB (0 dB would mean magnitude == 1 in this raw scale).
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    return 20.0 * np.log10(mag + epsilon)


def get_freq_bins(sample_rate: int, chunk_size: int) -> np.ndarray:
    """
    Map each rFFT bin index to a physical frequency in hertz.

    Bin spacing (frequency resolution) is ``sample_rate / chunk_size`` Hz:
    longer frames or lower sample rates give finer resolution but slower
    time updates.

    Parameters
    ----------
    sample_rate : int
        Samples per second (Hz).
    chunk_size : int
        FFT length / frame size in samples.

    Returns
    -------
    np.ndarray
        Frequency in Hz for each rFFT bin, from 0 to Nyquist (sample_rate/2).
    """
    return np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)


def analyze_frame(
    samples: np.ndarray,
    sample_rate: int,
    chunk_size: int,
    window_type: str = WINDOW_TYPE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full spectrum pipeline on one audio chunk.

    Parameters
    ----------
    samples : np.ndarray
        Time-domain audio, length ``chunk_size``.
    sample_rate : int
        Sample rate in Hz.
    chunk_size : int
        Expected frame length (used for frequency axis).
    window_type : str
        Window name passed to :func:`apply_window`.

    Returns
    -------
    freq_hz : np.ndarray
        Frequency axis for each bin.
    magnitude_db : np.ndarray
        Spectrum in dB for each bin.
    """
    windowed = apply_window(samples, window_type)
    magnitude = compute_fft_magnitude(windowed)
    magnitude_db = magnitude_to_db(magnitude)
    freq_hz = get_freq_bins(sample_rate, chunk_size)
    return freq_hz, magnitude_db
