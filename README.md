$$
\Huge \textit{Real-Time Audio Spectrum Analyser}
$$

<p align="center">
<em>Live microphone or system-audio capture → FFT → real-time bar chart or scrolling waterfall spectrogram, with optional pitch detection.</em>
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

This tool captures live audio — from a microphone or, on supported platforms, system-audio **loopback** — computes a one-sided FFT spectrum in real time, and renders it as a **bar chart** or a **scrolling waterfall spectrogram**. An optional autocorrelation-based **pitch tracker** overlays the detected note name, frequency, and cents deviation on either view. A calibration workflow using a 1 kHz test tone lets you verify frequency accuracy before trusting the display.

**Stack:** Python 3.9+, `sounddevice` (PyAudio fallback for mic-only capture), `numpy`, `matplotlib`.

---

$$
\Large \textit{Prerequisites}
$$

- Python **3.9+**
- A working microphone (built-in, headset, or USB), and/or a loopback-capable output device for system-audio capture
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

`sounddevice` is the primary backend. If it's unavailable on your system, install PyAudio manually — note that **loopback capture requires `sounddevice`** and has no PyAudio equivalent:

```bash
pip install PyAudio
```

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

**Pitch overlay** — adds detected note name, frequency, and cents offset to either view:

```bash
python main.py --pitch
python main.py --mode waterfall --pitch
```

**Capture source** — microphone (default) or system-audio loopback:

```bash
python main.py --source mic
python main.py --source loopback --device 4
```

```bash
python main.py --list-devices     # list input devices and loopback candidates
```

> On Windows, loopback works automatically off the default output device. On Linux/macOS you must pass an explicit `--device` — use `--list-devices` to find a PulseAudio/PipeWire `Monitor of ...` input (Linux) or a virtual device such as BlackHole (macOS).

Close the plot window or press **Ctrl+C** in the terminal to exit. The mic/loopback stream is always closed cleanly on exit.

---

$$
\Large \textit{Calibration}
$$

Verify frequency accuracy with a 1 kHz test tone.

**Terminal 1** — play a sine tone (default 1000 Hz, 30 s):

```bash
python test_tone.py
python test_tone.py --freq 1000 --duration 30 --amplitude 0.4
```

**Terminal 2** — capture a single-frame bar chart:

```bash
python main.py --static-bars
```

The tallest bar should land on the bin nearest **1000 Hz**, i.e. within one bin width. The DFT's frequency resolution is:

$$
\Delta f = \frac{f_s}{N}
$$

where $f_s$ is `SAMPLE_RATE` and $N$ is `CHUNK_SIZE` — at the defaults, $\Delta f = 44100 / 1024 \approx 43.07\ \text{Hz}$.

---

$$
\Large \textit{How It Works}
$$

Each captured frame of $N$ samples is windowed, transformed, and converted to decibels:

$$
X[k] = \sum_{n=0}^{N-1} w[n]\, x[n]\, e^{-i 2\pi kn/N}, \qquad k = 0, 1, \dots, \left\lfloor \frac{N}{2} \right\rfloor
$$

$$
\text{dB}[k] = 20 \log_{10}\!\big(|X[k]| + \varepsilon\big)
$$

where $w[n]$ is the selected window function (Hann, Hamming, or rectangular) and $\varepsilon = 10^{-6}$ avoids $\log(0)$. Only the one-sided (non-negative frequency) half of the spectrum is kept, via `numpy.fft.rfft`.

**Pitch detection** estimates the fundamental frequency from the frame's autocorrelation, computed efficiently in the frequency domain (Wiener–Khinchin theorem):

### Autocorrelation Formulation

$$
r[\tau] = \mathcal{F}^{-1}\left\lbrace \mathcal{F}\lbrace x\rbrace \cdot \overline{\mathcal{F}\lbrace x\rbrace} \right\rbrace, \qquad r[\tau] \leftarrow \frac{r[\tau]}{r[0]}
$$


The tracker searches for the highest peak of $r[\tau]$ within the lag range corresponding to `PITCH_FMIN`–`PITCH_FMAX`, refines it with parabolic interpolation for sub-bin accuracy, and accepts it only if the peak confidence exceeds `PITCH_CONFIDENCE_MIN` and the frame's RMS exceeds `PITCH_RMS_MIN`. The frequency is converted to a MIDI note number and cents offset via:

$$
m = 69 + 12 \log_2\!\left(\frac{f}{440}\right)
$$

A median filter over the last `PITCH_SMOOTHING_FRAMES` estimates smooths the displayed pitch.

---

$$
\Large \textit{Other Commands}
$$

| Command | Purpose |
|---|---|
| `python main.py --capture-test` | Print min/max/mean amplitude per chunk for 5 s |
| `python main.py --static-bars` | One-shot dB bar spectrum |
| `python main.py --live-bars` | Same as `--mode bars` |
| `python main.py --list-devices` | List audio devices and loopback candidates |

---

$$
\Large \textit{Configuration}
$$

Edit `config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `SAMPLE_RATE` | `44100` | Sample rate (Hz) |
| `CHUNK_SIZE` | `1024` | Samples per FFT frame |
| `CHANNELS` | `1` | Capture channel count |
| `WINDOW_TYPE` | `"hann"` | Default FFT window |
| `DB_MIN` / `DB_MAX` | `-60` / `0` | Display range (dB) |
| `WATERFALL_HISTORY` | `200` | Scrolling time depth (frames) |
| `COLORMAP` | `"inferno"` | Waterfall colormap |
| `TARGET_FPS` | `30` | Animation target |
| `PITCH_FMIN` / `PITCH_FMAX` | `60` / `1200` Hz | Search range for pitch detection |
| `PITCH_CONFIDENCE_MIN` | `0.45` | Min. normalized autocorrelation peak to report a pitch |
| `PITCH_RMS_MIN` | `0.01` | Frames quieter than this are treated as silence |
| `PITCH_SMOOTHING_FRAMES` | `5` | Median-filter window for the pitch overlay |
| `DEFAULT_SOURCE` | `"mic"` | Default `--source` (`mic` or `loopback`) |

---

$$
\Large \textit{Project Layout}
$$

```
spectrum_analyzer/
├── audio_capture.py   # Mic/loopback stream (sounddevice / PyAudio)
├── dsp.py             # Window, rFFT, dB, frequency bins
├── pitch_detect.py    # Autocorrelation pitch tracker + note naming
├── visualizer.py       # Static bars, live bars, waterfall, pitch overlay
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
| Loopback unavailable (no device given, or platform limitation) | `LoopbackUnavailableError` with platform-specific setup hints and device candidates |
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

- Export spectrogram / recording to file
- Web Audio browser UI

---

<p align="center"><em>MIT License</em></p>
