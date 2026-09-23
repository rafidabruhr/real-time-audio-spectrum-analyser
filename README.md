$$
\Huge \textit{Real-Time Audio Spectrum Analyser}
$$

<p align="center">
<em>Live microphone capture → FFT → real-time bar chart or scrolling waterfall spectrogram, in pure Python.</em>
</p>

<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-3.9%2B-blue">
<img alt="license" src="https://img.shields.io/badge/license-MIT-green">
<img alt="status" src="https://img.shields.io/badge/status-active-brightgreen">
</p>

---

$$
\LARGE \textit{Overview}
$$

This tool captures live audio from your microphone, computes a one-sided FFT spectrum in real time, and renders it as either a **bar chart** or a **scrolling waterfall spectrogram**. It also includes a calibration workflow using a 1 kHz test tone, so you can verify frequency accuracy before trusting the display.

**Stack:** Python 3.9+, `sounddevice` (PyAudio fallback), `numpy`, `matplotlib`.

---

$$
\Large \textit{Prerequisites}
$$

- Python **3.9+**
- A working microphone (built-in, headset, or USB)
- **macOS / Windows:** allow microphone access when prompted
- **Linux:** PulseAudio or PipeWire (default on Fedora/Ubuntu)

If `sounddevice` fails to install, you may need the PortAudio system package:

| Platform | Package |
|---|---|
| Fedora | `portaudio-devel` |
| Debian/Ubuntu | `libportaudio2`, `portaudio19-dev` |

---

$$
\Large \textit{Installation}
$$

```bash
cd spectrum_analyzer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `sounddevice` is unavailable on your system, install PyAudio manually:

```bash
pip install PyAudio
```

The app auto-detects the available backend.

---

$$
\Large \textit{Quick Start}
$$

**Bar spectrum (default):**

```bash
python main.py
python main.py --mode bars
```

**Scrolling waterfall** (200-frame history, `inferno` colourmap — brighter = louder):

```bash
python main.py --mode waterfall
```

**FFT window** (affects spectral leakage on both views):

```bash
python main.py --mode waterfall --window hann
python main.py --window hamming
python main.py --window rectangular
```

| Window | Effect |
|---|---|
| `hann` | Default; good leakage vs. resolution tradeoff |
| `hamming` | Similar to Hann; slightly different sidelobe floor |
| `rectangular` | Strongest sidelobes — smeared skirts on waterfall; useful for comparing leakage in lab settings |

Close the plot window or press **Ctrl+C** in the terminal to exit. The mic stream is always closed cleanly on exit.

---

$$
\Large \textit{Calibration}
$$

Verify frequency accuracy with a 1 kHz test tone.

**Terminal 1** — play a sine tone (default 1000 Hz):

```bash
python test_tone.py
python test_tone.py --freq 1000 --duration 60 --amplitude 0.3
```

**Terminal 2** — capture a single-frame bar chart:

```bash
python main.py --static-bars
```

The tallest bar should land on the bin nearest **1000 Hz**, i.e. within one bin width, where the frequency resolution is:
```math
\Delta f = \frac{\text{SAMPLE\_RATE}}{\text{CHUNK\_SIZE}}
```

---

$$
\Large \textit{Other Commands}
$$

| Command | Purpose |
|---|---|
| `python main.py --capture-test` | Phase 1: print min/max/mean amplitude per chunk for 5 s |
| `python main.py --static-bars` | One-shot dB bar spectrum |
| `python main.py --live-bars` | Same as `--mode bars` |

---

$$
\Large \textit{Configuration}
$$

Edit `config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `SAMPLE_RATE` | `44100` | Sample rate (Hz) |
| `CHUNK_SIZE` | `1024` | Samples per FFT frame |
| `DB_MIN` / `DB_MAX` | `-60` / `0` | Display range (dB) |
| `WATERFALL_HISTORY` | `200` | Scrolling time depth (frames) |
| `COLORMAP` | `inferno` | Waterfall colormap |
| `TARGET_FPS` | `30` | Animation target |

---

$$
\Large \textit{Project Layout}
$$

```
spectrum_analyzer/
├── audio_capture.py   # Mic stream (sounddevice / PyAudio)
├── dsp.py             # Window, rFFT, dB, frequency bins
├── visualizer.py      # Static bars, live bars, waterfall
├── config.py           # Tunables
├── main.py             # CLI entry point
├── test_tone.py        # Calibration sine playback
├── requirements.txt
└── README.md
```

---

$$
\Large \textit{Error Handling}
$$

The app tries to fail gracefully:

| Situation | What you see |
|---|---|
| No mic / no backend | Clear message with permissions, wiring, and install hints |
| Device unplugged mid-run | `AudioStreamReadError` with reconnect hint |
| Buffer overflow / underrun | One-time warning; capture may continue |
| Plot window closed | Animation stops, stream closed |
| Ctrl+C | Same clean shutdown |

---

$$
\Large \textit{Performance Notes}
$$

Live views use **blitting** (bars) or **`imshow.set_data`** (waterfall) to avoid redrawing the full figure each frame, targeting **30 fps** (`TARGET_FPS` in `config.py`). On slower machines, try closing other apps, or swap `matplotlib` for `pyqtgraph` (commented out in `visualizer.py`).

---

$$
\Large \textit{Stretch Goals}
$$

*Not implemented unless requested:*

- Pitch detection
- System loopback input
- Export spectrogram/recording
- Web Audio browser UI

---

<p align="center"><em>MIT License</em></p>
