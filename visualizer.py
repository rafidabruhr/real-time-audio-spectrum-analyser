import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

from audio_capture import AudioCaptureError, AudioStream, AudioStreamReadError
from config import (
    CHUNK_SIZE, COLORMAP, DB_MAX, DB_MIN, SAMPLE_RATE, TARGET_FPS,
    WATERFALL_HISTORY, WINDOW_TYPE,
)
from dsp import analyze_frame, get_freq_bins
from pitch_detect import PitchTracker


class WaterfallBuffer:
    def __init__(self, history_len, num_bins, fill_value=DB_MIN):
        self.history_len = history_len
        self.num_bins = num_bins
        self._data = np.full((history_len, num_bins), fill_value, dtype=np.float32)

    @property
    def data(self):
        return self._data

    def push_row(self, spectrum_db):
        # scroll in place instead of allocating a new array every frame
        self._data[:-1, :] = self._data[1:, :]
        self._data[-1, :] = spectrum_db
        return self._data


def _ensure_chunk_length(samples):
    if len(samples) == CHUNK_SIZE:
        return samples
    if len(samples) > CHUNK_SIZE:
        return samples[:CHUNK_SIZE]
    return np.concatenate([samples, np.zeros(CHUNK_SIZE - len(samples), dtype=samples.dtype)])


def _open_mic_stream(source="mic", device=None):
    try:
        stream = AudioStream(source=source, device=device)
        stream.read_chunk()  # make sure it actually works before we hand it back
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    return stream


def _format_frequency_hz(value, _pos):
    return f"{value / 1000.0:.1f} kHz" if value >= 1000.0 else f"{int(value)} Hz"


def _pitch_label(estimate):
    if estimate.frequency_hz is None or estimate.note_name is None:
        return ""
    sign = "+" if estimate.cents_off >= 0 else ""
    return f"{estimate.note_name}  {estimate.frequency_hz:.1f} Hz  ({sign}{estimate.cents_off:.0f}¢)"


def _read_chunk_or_stop(stream, closed, shutdown, fig):
    try:
        return _ensure_chunk_length(stream.read_chunk())
    except (AudioStreamReadError, AudioCaptureError) as err:
        closed["flag"] = True
        print(f"\n{err}", file=sys.stderr)
        shutdown()
        plt.close(fig)
        return None


def run_static_bar_spectrum(window_type=WINDOW_TYPE, show_pitch=False, source="mic", device=None):
    # delta_f = sample_rate / chunk_size, ~43 Hz/bin at 44.1kHz + 1024 samples
    bin_width_hz = SAMPLE_RATE / CHUNK_SIZE

    try:
        with AudioStream(source=source, device=device) as stream:
            stream.read_chunk()  # throwaway, let levels settle
            samples = stream.read_chunk()
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    samples = _ensure_chunk_length(samples)
    freq_hz, mag_db = analyze_frame(samples, SAMPLE_RATE, CHUNK_SIZE, window_type)
    mag_db = np.clip(mag_db, DB_MIN, DB_MAX)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(freq_hz, mag_db, width=bin_width_hz * 0.9, align="center")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    title = f"Single-frame spectrum (window={window_type}, Δf = {bin_width_hz:.2f} Hz, source={source})"

    if show_pitch:
        estimate = PitchTracker().update(samples, SAMPLE_RATE)
        label = _pitch_label(estimate)
        if label:
            title += f"\nPitch: {label}"
            if estimate.frequency_hz is not None:
                ax.axvline(estimate.frequency_hz, color="crimson", linestyle="--", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_ylim(DB_MIN, DB_MAX)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def run_live_bar_spectrum(window_type=WINDOW_TYPE, show_pitch=False, source="mic", device=None):
    bin_width_hz = SAMPLE_RATE / CHUNK_SIZE
    freq_hz = get_freq_bins(SAMPLE_RATE, CHUNK_SIZE)
    interval_ms = max(1, int(1000 / TARGET_FPS))

    stream = _open_mic_stream(source, device)
    pitch_tracker = PitchTracker() if show_pitch else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"Live spectrum (window={window_type}, Δf = {bin_width_hz:.2f} Hz, source={source})")
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_ylim(DB_MIN, DB_MAX)
    ax.grid(True, alpha=0.3)

    bars = ax.bar(freq_hz, np.full(len(freq_hz), DB_MIN), width=bin_width_hz * 0.9,
                   align="center", color="steelblue", edgecolor="none")
    patches = list(bars.patches)
    for p in patches:
        p.set_animated(True)

    pitch_line = pitch_text = None
    if show_pitch:
        pitch_line = ax.axvline(SAMPLE_RATE / 4, color="crimson", linestyle="--", linewidth=1.5, visible=False)
        pitch_line.set_animated(True)
        pitch_text = ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top",
                              fontsize=12, color="crimson",
                              bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.85))
        pitch_text.set_animated(True)

    fig.tight_layout()
    fig.canvas.draw()

    anim = None
    closed = {"flag": False}

    def shutdown():
        if anim is not None and anim.event_source is not None:
            anim.event_source.stop()
        stream.close()

    def on_close(_event):
        closed["flag"] = True
        shutdown()

    def on_sigint(_signum, _frame):
        closed["flag"] = True
        shutdown()
        plt.close(fig)

    fig.canvas.mpl_connect("close_event", on_close)
    signal.signal(signal.SIGINT, on_sigint)

    def artists():
        extra = [a for a in (pitch_line, pitch_text) if a is not None]
        return patches + extra

    def update(_frame):
        if closed["flag"]:
            return artists()

        samples = _read_chunk_or_stop(stream, closed, shutdown, fig)
        if samples is None:
            return artists()

        _, mag_db = analyze_frame(samples, SAMPLE_RATE, CHUNK_SIZE, window_type)
        mag_db = np.clip(mag_db, DB_MIN, DB_MAX)
        for rect, height in zip(patches, mag_db):
            rect.set_height(height)

        if show_pitch:
            estimate = pitch_tracker.update(samples, SAMPLE_RATE)
            if estimate.frequency_hz is not None:
                pitch_line.set_xdata([estimate.frequency_hz, estimate.frequency_hz])
                pitch_line.set_visible(True)
                pitch_text.set_text(_pitch_label(estimate))
            else:
                pitch_line.set_visible(False)
                pitch_text.set_text("")

        return artists()

    anim = FuncAnimation(fig, update, interval=interval_ms, blit=True, cache_frame_data=False)

    try:
        plt.show()
    finally:
        closed["flag"] = True
        shutdown()


def run_live_waterfall(window_type=WINDOW_TYPE, show_pitch=False, source="mic", device=None):
    num_bins = CHUNK_SIZE // 2 + 1
    nyquist = SAMPLE_RATE / 2.0
    history_sec = WATERFALL_HISTORY * (CHUNK_SIZE / SAMPLE_RATE)
    waterfall = WaterfallBuffer(WATERFALL_HISTORY, num_bins, fill_value=DB_MIN)
    closed = {"flag": False}

    stream = _open_mic_stream(source, device)
    pitch_tracker = PitchTracker() if show_pitch else None

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(waterfall.data, aspect="auto", origin="upper", cmap=COLORMAP,
                    vmin=DB_MIN, vmax=DB_MAX, extent=(0.0, nyquist, 0.0, history_sec),
                    interpolation="nearest", animated=True)
    imgs = [im]
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Live waterfall (window={window_type}, source={source})")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_frequency_hz))
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Level (dB)")

    pitch_line = pitch_text = None
    if show_pitch:
        pitch_line = ax.axvline(nyquist / 2, color="cyan", linestyle="--", linewidth=1.5, visible=False)
        pitch_line.set_animated(True)
        pitch_text = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                              fontsize=12, color="black",
                              bbox=dict(boxstyle="round", fc="cyan", ec="black", alpha=0.85))
        pitch_text.set_animated(True)
        imgs += [pitch_line, pitch_text]

    fig.tight_layout()
    fig.canvas.draw()

    anim_ref = [None]

    def shutdown():
        if anim_ref[0] is not None and anim_ref[0].event_source is not None:
            anim_ref[0].event_source.stop()
        stream.close()

    def on_close(_event):
        closed["flag"] = True
        shutdown()

    def on_sigint(_signum, _frame):
        closed["flag"] = True
        shutdown()
        plt.close(fig)

    fig.canvas.mpl_connect("close_event", on_close)
    signal.signal(signal.SIGINT, on_sigint)

    def update(_frame):
        if closed["flag"]:
            return imgs

        samples = _read_chunk_or_stop(stream, closed, shutdown, fig)
        if samples is None:
            return imgs

        _, mag_db = analyze_frame(samples, SAMPLE_RATE, CHUNK_SIZE, window_type)
        im.set_data(waterfall.push_row(np.clip(mag_db, DB_MIN, DB_MAX)))

        if show_pitch:
            estimate = pitch_tracker.update(samples, SAMPLE_RATE)
            if estimate.frequency_hz is not None:
                pitch_line.set_xdata([estimate.frequency_hz, estimate.frequency_hz])
                pitch_line.set_visible(True)
                pitch_text.set_text(_pitch_label(estimate))
            else:
                pitch_line.set_visible(False)
                pitch_text.set_text("")

        return imgs

    interval_ms = max(1, int(1000 / TARGET_FPS))
    anim_ref[0] = FuncAnimation(fig, update, interval=interval_ms, blit=True, cache_frame_data=False)

    try:
        plt.show()
    finally:
        closed["flag"] = True
        shutdown()


if __name__ == "__main__":
    run_live_bar_spectrum()