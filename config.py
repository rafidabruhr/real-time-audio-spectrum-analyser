"""
Central configuration for the real-time spectrum analyzer.

All tunable constants live here so DSP, capture, and visualization stay
in sync without magic numbers scattered through the codebase.
"""

SAMPLE_RATE: int = 44100
CHUNK_SIZE: int = 1024  # samples per buffer (one FFT frame later)
CHANNELS: int = 1
WINDOW_TYPE: str = "hann"  # supported: "hann", "hamming", "rectangular"
DB_MIN: float = -60.0  # display floor for dB scale
DB_MAX: float = 0.0
WATERFALL_HISTORY: int = 200  # past frames in scrolling spectrogram
COLORMAP: str = "inferno"
TARGET_FPS: int = 30  # minimum goal for live matplotlib views

# --- Pitch overlay (extension) ---
PITCH_FMIN: float = 60.0            # Hz, ~B1; below typical bass voice/instrument range
PITCH_FMAX: float = 1200.0          # Hz, covers voice + most instrument fundamentals
PITCH_CONFIDENCE_MIN: float = 0.45  # normalized autocorrelation peak required to report a pitch
PITCH_RMS_MIN: float = 0.01         # frames quieter than this are treated as silence
PITCH_SMOOTHING_FRAMES: int = 5     # median-filter window for the overlay display

# --- Loopback capture (extension) ---
DEFAULT_SOURCE: str = "mic"  # "mic" or "loopback" ("what you hear" / system audio)
