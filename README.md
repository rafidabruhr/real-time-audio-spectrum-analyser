# Real-Time Audio Spectrum Analyzer

Educational DSP project: capture live microphone audio, compute a one-sided FFT spectrum, and display it as a **real-time bar chart** or **scrolling waterfall spectrogram**.

Stack: Python 3.9+, `sounddevice` (PyAudio fallback), `numpy`, `matplotlib`.

---

## Prerequisites

- Python **3.9+**
- Working microphone (built-in, headset, or USB)
- **macOS / Windows:** allow microphone access when prompted
- **Linux:** PulseAudio or PipeWire (Fedora/Ubuntu defaults)

Optional system package for PortAudio (if `sounddevice` fails to install):  
`portaudio-devel` (Fedora), `libportaudio2` / `portaudio19-dev` (Debian/Ubuntu).

---

## Install

```bash
cd spectrum_analyzer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `sounddevice` is unavailable, install PyAudio manually (`pip install PyAudio`) — the app detects either backend.

---

## Quick start (live analyzer)

**Bar spectrum (default):**

```bash
python main.py
python main.py --mode bars
```

**Scrolling waterfall** (200-frame history, `inferno` colormap — brighter = louder):

```bash
python main.py --mode waterfall
```

**FFT window** (affects spectral leakage on both views):

```bash
python main.py --mode waterfall --window hann
python main.py --window hamming
python main.py --window rectangular
```

| Window        | Visual / audible effect |
|---------------|-------------------------|
| `hann`        | Default; good leakage vs resolution tradeoff |
| `hamming`     | Similar to Hann; slightly different sidelobe floor |
| `rectangular` | Strongest sidelobes — smeared skirts on waterfall, useful for comparing leakage in lab |

Close the plot window or press **Ctrl+C** in the terminal to exit. The mic stream is always closed on exit.

---

## Calibration (1 kHz test tone)

**Terminal 1** — play a sine tone (default 1000 Hz):

```bash
python test_tone.py
python test_tone.py --freq 1000 --duration 60 --amplitude 0.3
```

**Terminal 2** — single-frame bar chart:

```bash
python main.py --static-bars
```

The tallest bar should land on the bin nearest **1000 Hz**. Frequency resolution:

\[
\Delta f = \frac{\text{SAMPLE\_RATE}}{\text{CHUNK\_SIZE}} = \frac{44100}{1024} \approx 43.07\ \text{Hz}
\]

Peak must be within **one bin width** of 1000 Hz.

---

## Other commands

| Command | Purpose |
|---------|---------|
| `python main.py --capture-test` | Phase 1: print min/max/mean amplitude per chunk for 5 s |
| `python main.py --static-bars` | One-shot dB bar spectrum |
| `python main.py --live-bars` | Same as `--mode bars` |

---

## Configuration

Edit `config.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `SAMPLE_RATE` | 44100 | Hz |
| `CHUNK_SIZE` | 1024 | Samples per FFT frame |
| `DB_MIN` / `DB_MAX` | -60 / 0 | Display range (dB) |
| `WATERFALL_HISTORY` | 200 | Scrolling time depth (frames) |
| `COLORMAP` | inferno | Waterfall colormap |
| `TARGET_FPS` | 30 | Animation target |

---

## Project layout

```
spectrum_analyzer/
├── audio_capture.py   # Mic stream (sounddevice / PyAudio)
├── dsp.py             # Window, rFFT, dB, frequency bins
├── visualizer.py      # Static bars, live bars, waterfall
├── config.py          # Tunables
├── main.py            # CLI entry point
├── test_tone.py       # Calibration sine playback
├── requirements.txt
└── README.md
```

---

## Error messages (no raw tracebacks)

The app tries to fail gracefully:

| Situation | What you see |
|-----------|----------------|
| No mic / no backend | Clear message: permissions, plug in device, install deps |
| Device unplugged mid-run | `AudioStreamReadError` with reconnect hint |
| Buffer overflow / underrun | One-time warning; capture may continue |
| Close plot window | Animation stops, stream closed |
| Ctrl+C | Same clean shutdown |

---

## Performance notes

Live views use **blitting** (bars) or **`imshow.set_data`** (waterfall) to avoid redrawing the full figure. Target **30 fps** (`TARGET_FPS` in `config.py`). If a laptop struggles, try closing other apps or swap matplotlib for **pyqtgraph** (commented in `visualizer.py`).

---

## Stretch goals (not implemented unless requested)

Pitch detection, system loopback input, export spectrogram/recording, Web Audio browser UI.
