from dataclasses import dataclass
import numpy as np
from config import PITCH_CONFIDENCE_MIN, PITCH_FMAX, PITCH_FMIN, PITCH_RMS_MIN

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
A4_HZ = 440.0
A4_MIDI = 69


@dataclass
class PitchEstimate:
    frequency_hz: float | None
    confidence: float
    note_name: str | None
    cents_off: float


def hz_to_note(freq):
    midi = A4_MIDI + 12.0 * np.log2(freq / A4_HZ)
    nearest = round(midi)
    cents = (midi - nearest) * 100.0
    return f"{NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}", cents


def autocorrelation_pitch(samples, sample_rate, fmin=PITCH_FMIN, fmax=PITCH_FMAX):
    x = np.asarray(samples, dtype=np.float64)
    x -= x.mean()

    rms = np.sqrt(np.mean(x * x))
    if rms < PITCH_RMS_MIN:
        return None, 0.0

    n = len(x)
    # FFT-based autocorrelation is way faster than a naive O(n^2) loop here
    fft_size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, n=fft_size)
    acf = np.fft.irfft(spec * np.conj(spec), n=fft_size)[:n]

    if acf[0] <= 0:
        return None, 0.0
    acf /= acf[0]

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

    # parabolic interpolation for sub-sample lag accuracy
    if 0 < peak_lag < n - 1:
        y0, y1, y2 = acf[peak_lag - 1], acf[peak_lag], acf[peak_lag + 1]
        denom = y0 - 2 * y1 + y2
        shift = np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0) if denom else 0.0
    else:
        shift = 0.0

    freq = sample_rate / (peak_lag + shift)
    return freq, confidence


class PitchTracker:
    def __init__(self, history=5):
        self.history = history
        self.recent = []

    def update(self, samples, sample_rate):
        freq, conf = autocorrelation_pitch(samples, sample_rate)
        if freq is None:
            self.recent.clear()
            return PitchEstimate(None, conf, None, 0.0)

        self.recent.append(freq)
        if len(self.recent) > self.history:
            self.recent.pop(0)

        smoothed = float(np.median(self.recent))
        note, cents = hz_to_note(smoothed)
        return PitchEstimate(smoothed, conf, note, cents)