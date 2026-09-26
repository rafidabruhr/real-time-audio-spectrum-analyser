import numpy as np
from config import WINDOW_TYPE


def apply_window(samples, window_type=WINDOW_TYPE):
    x = np.asarray(samples, dtype=np.float64)
    n = len(x)
    wtype = window_type.lower()

    if wtype == "hann":
        win = np.hanning(n)
    elif wtype == "hamming":
        win = np.hamming(n)
    elif wtype == "rectangular":
        win = np.ones(n)
    else:
        raise ValueError(f"unknown window_type: {window_type}")

    return x * win


def compute_fft_magnitude(windowed):
    return np.abs(np.fft.rfft(windowed))


def magnitude_to_db(magnitude, epsilon=1e-6):
    return 20.0 * np.log10(np.asarray(magnitude, dtype=np.float64) + epsilon)


def get_freq_bins(sample_rate, chunk_size):
    return np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)


def analyze_frame(samples, sample_rate, chunk_size, window_type=WINDOW_TYPE):
    windowed = apply_window(samples, window_type)
    mag = compute_fft_magnitude(windowed)
    return get_freq_bins(sample_rate, chunk_size), magnitude_to_db(mag)