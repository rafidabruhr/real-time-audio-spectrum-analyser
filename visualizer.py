from __future__ import annotations

import signal
import sys
from typing import Callable, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

from audio_capture import (
    AudioCaptureError,
    AudioStream,
    AudioStreamReadError,
    InputSource,
)
from config import (
    CHUNK_SIZE,
    COLORMAP,
    DB_MAX,
    DB_MIN,
    SAMPLE_RATE,
    TARGET_FPS,
    WATERFALL_HISTORY,
    WINDOW_TYPE,
)
from dsp import analyze_frame, get_freq_bins
from pitch_detect import PitchEstimate, PitchTracker


class WaterfallBuffer:
    def __init__(
        self,
        history_len: int,
        num_bins: int,
        fill_value: float = DB_MIN,
    ) -> None:
        self.history_len = history_len
        self.num_bins = num_bins
        self._data = np.full(
            (history_len, num_bins), fill_value, dtype=np.float32
        )

    @property
    def data(self) -> np.ndarray:
        """Current waterfall matrix for rendering (time × frequency)."""
        return self._data

    def push_row(self, spectrum_db: np.ndarray) -> np.ndarray:
        # In-place scroll: oldest slice falls off the top, no new array allocated.
        self._data[0:-1, :] = self._data[1:, :]
        self._data[-1, :] = spectrum_db.astype(np.float32, copy=False)
        return self._data


def _ensure_chunk_length(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] == CHUNK_SIZE:
        return samples
    if samples.shape[0] > CHUNK_SIZE:
        return samples[:CHUNK_SIZE]
    pad = np.zeros(CHUNK_SIZE - samples.shape[0], dtype=samples.dtype)
    return np.concatenate([samples, pad])


def _open_mic_stream(
    source: InputSource = "mic",
    device: Optional[int] = None,
) -> AudioStream:
    try:
        stream = AudioStream(source=source, device=device)
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    try:
        _ = stream.read_chunk()
    except AudioCaptureError as err:
        stream.close()
        print(err, file=sys.stderr)
        sys.exit(1)

    return stream


def _format_frequency_hz(value: float, _pos: int) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.1f} kHz"
    return f"{int(value)} Hz"


def _pitch_label(estimate: PitchEstimate) -> str:
    if estimate.frequency_hz is None or estimate.note_name is None:
        return ""
    sign = "+" if estimate.cents_off >= 0 else ""
    return (
        f"{estimate.note_name}  {estimate.frequency_hz:.1f} Hz  "
        f"({sign}{estimate.cents_off:.0f}¢)"
    )


def _read_chunk_or_stop(
    stream: AudioStream,
    closed: dict,
    shutdown: Callable[[], None],
    fig: Figure,
) -> Optional[np.ndarray]:
    try:
        return _ensure_chunk_length(stream.read_chunk())
    except AudioStreamReadError as err:
        closed["flag"] = True
        print(f"\n{err}", file=sys.stderr)
        shutdown()
        plt.close(fig)
        return None
    except AudioCaptureError as err:
        closed["flag"] = True
        print(f"\n{err}", file=sys.stderr)
        shutdown()
        plt.close(fig)
        return None


def run_static_bar_spectrum(
    window_type: str = WINDOW_TYPE,
    show_pitch: bool = False,
    source: InputSource = "mic",
    device: Optional[int] = None,
) -> None:
    # Frequency resolution of the DFT: delta_f = sample_rate / chunk_size.
    # At 44100 Hz and N=1024, delta_f = 44100/1024 ≈ 43.07 Hz per bin.
    # A 1000 Hz tone peak should lie on the bin nearest 1000 Hz, within ±delta_f.
    bin_width_hz = SAMPLE_RATE / CHUNK_SIZE

    try:
        with AudioStream(source=source, device=device) as stream:
            _ = stream.read_chunk()
            samples = stream.read_chunk()
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    samples = _ensure_chunk_length(samples)
    freq_hz, magnitude_db = analyze_frame(
        samples, SAMPLE_RATE, CHUNK_SIZE, window_type=window_type
    )
    magnitude_db = np.clip(magnitude_db, DB_MIN, DB_MAX)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(freq_hz, magnitude_db, width=bin_width_hz * 0.9, align="center")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    title = (
        f"Single-frame spectrum (window={window_type}, "
        f"Δf = {bin_width_hz:.2f} Hz, source={source})"
    )

    if show_pitch:
        tracker = PitchTracker()
        estimate = tracker.update(samples, SAMPLE_RATE)
        label = _pitch_label(estimate)
        if label:
            title += f"\nPitch: {label}"
            peak_hz = estimate.frequency_hz
            if peak_hz is not None:
                ax.axvline(peak_hz, color="crimson", linestyle="--", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_ylim(DB_MIN, DB_MAX)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def run_live_bar_spectrum(
    window_type: str = WINDOW_TYPE,
    show_pitch: bool = False,
    source: InputSource = "mic",
    device: Optional[int] = None,
) -> None:
    bin_width_hz = SAMPLE_RATE / CHUNK_SIZE
    freq_hz = get_freq_bins(SAMPLE_RATE, CHUNK_SIZE)
    num_bins = freq_hz.shape[0]
    interval_ms = max(1, int(1000 / TARGET_FPS))

    stream = _open_mic_stream(source=source, device=device)
    pitch_tracker = PitchTracker() if show_pitch else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(
        f"Live spectrum (window={window_type}, Δf = {bin_width_hz:.2f} Hz, "
        f"source={source})"
    )
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_ylim(DB_MIN, DB_MAX)
    ax.grid(True, alpha=0.3)

    initial_heights = np.full(num_bins, DB_MIN, dtype=np.float64)
    bar_container = ax.bar(
        freq_hz,
        initial_heights,
        width=bin_width_hz * 0.9,
        align="center",
        color="steelblue",
        edgecolor="none",
    )
    patches: List[Rectangle] = list(bar_container.patches)
    for patch in patches:
        patch.set_animated(True)

    pitch_line: Optional[Line2D] = None
    pitch_text: Optional[Text] = None
    if show_pitch:
        pitch_line = ax.axvline(
            SAMPLE_RATE / 4, color="crimson", linestyle="--", linewidth=1.5,
            visible=False,
        )
        pitch_line.set_animated(True)
        pitch_text = ax.text(
            0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color="crimson",
            bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.85),
        )
        pitch_text.set_animated(True)

    fig.tight_layout()
    fig.canvas.draw()

    anim: Optional[FuncAnimation] = None
    closed = {"flag": False}

    def _shutdown() -> None:
        if anim is not None:
            event_source = getattr(anim, "event_source", None)
            if event_source is not None:
                event_source.stop()
        stream.close()

    def _on_close(_event) -> None:
        closed["flag"] = True
        _shutdown()

    def _on_sigint(_signum, _frame) -> None:
        closed["flag"] = True
        _shutdown()
        plt.close(fig)

    fig.canvas.mpl_connect("close_event", _on_close)
    signal.signal(signal.SIGINT, _on_sigint)

    def _artists() -> List[Artist]:
        extra = [a for a in (pitch_line, pitch_text) if a is not None]
        return cast(List[Artist], patches + extra)

    def _update(_frame: int) -> List[Artist]:
        if closed["flag"]:
            return _artists()

        samples = _read_chunk_or_stop(stream, closed, _shutdown, fig)
        if samples is None:
            return _artists()

        _, magnitude_db = analyze_frame(
            samples, SAMPLE_RATE, CHUNK_SIZE, window_type=window_type
        )
        magnitude_db_clipped = np.clip(magnitude_db, DB_MIN, DB_MAX)

        for rect, height in zip(patches, magnitude_db_clipped):
            rect.set_height(float(height))

        if show_pitch and pitch_tracker is not None:
            line = pitch_line
            text = pitch_text
            assert line is not None
            assert text is not None
            estimate = pitch_tracker.update(samples, SAMPLE_RATE)
            if estimate.frequency_hz is not None:
                line.set_xdata([estimate.frequency_hz, estimate.frequency_hz])
                line.set_visible(True)
                text.set_text(_pitch_label(estimate))
            else:
                line.set_visible(False)
                text.set_text("")

        return _artists()

    anim = FuncAnimation(
        fig,
        _update,
        interval=interval_ms,
        blit=True,
        cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        closed["flag"] = True
        _shutdown()


def run_live_waterfall(
    window_type: str = WINDOW_TYPE,
    show_pitch: bool = False,
    source: InputSource = "mic",
    device: Optional[int] = None,
) -> None:
    num_bins = CHUNK_SIZE // 2 + 1
    nyquist = SAMPLE_RATE / 2.0
    frame_duration_sec = CHUNK_SIZE / SAMPLE_RATE
    history_duration_sec = WATERFALL_HISTORY * frame_duration_sec
    waterfall = WaterfallBuffer(WATERFALL_HISTORY, num_bins, fill_value=DB_MIN)
    image_holder: List[Artist] = []
    closed = {"flag": False}

    stream = _open_mic_stream(source=source, device=device)
    pitch_tracker = PitchTracker() if show_pitch else None

    fig, ax = plt.subplots(figsize=(10, 6))
    # Horizontal axis: 0 Hz → Nyquist. Vertical axis: time in seconds (newest at bottom).
    im = ax.imshow(
        waterfall.data,
        aspect="auto",
        origin="upper",
        cmap=COLORMAP,
        vmin=DB_MIN,
        vmax=DB_MAX,
        extent=(0.0, nyquist, 0.0, history_duration_sec),
        interpolation="nearest",
        animated=True,
    )
    image_holder.append(im)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Live waterfall (window={window_type}, source={source})")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_frequency_hz))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Level (dB)")

    pitch_line: Optional[Line2D] = None
    pitch_text: Optional[Text] = None
    if show_pitch:
        pitch_line = ax.axvline(
            nyquist / 2, color="cyan", linestyle="--", linewidth=1.5, visible=False,
        )
        pitch_line.set_animated(True)
        pitch_text = ax.text(
            0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color="black",
            bbox=dict(boxstyle="round", fc="cyan", ec="black", alpha=0.85),
        )
        pitch_text.set_animated(True)
        image_holder.extend([pitch_line, pitch_text])  # kept animated/blit-managed

    fig.tight_layout()
    fig.canvas.draw()

    def _shutdown(anim: Optional[FuncAnimation]) -> None:
        if anim is not None:
            event_source = getattr(anim, "event_source", None)
            if event_source is not None:
                event_source.stop()
        stream.close()

    anim_ref: List[Optional[FuncAnimation]] = [None]

    def _on_close(_event) -> None:
        closed["flag"] = True
        _shutdown(anim_ref[0])

    def _on_sigint(_signum, _frame) -> None:
        closed["flag"] = True
        _shutdown(anim_ref[0])
        plt.close(fig)

    def _shutdown_live() -> None:
        _shutdown(anim_ref[0])

    fig.canvas.mpl_connect("close_event", _on_close)
    signal.signal(signal.SIGINT, _on_sigint)

    def _update(_frame: int) -> List[Artist]:
        if closed["flag"]:
            return image_holder

        samples = _read_chunk_or_stop(stream, closed, _shutdown_live, fig)
        if samples is None:
            return image_holder

        _, magnitude_db = analyze_frame(
            samples, SAMPLE_RATE, CHUNK_SIZE, window_type=window_type
        )
        magnitude_db_clipped = np.clip(magnitude_db, DB_MIN, DB_MAX)
        matrix = waterfall.push_row(magnitude_db_clipped)
        im.set_data(matrix)

        if show_pitch and pitch_tracker is not None:
            line = pitch_line
            text = pitch_text
            assert line is not None
            assert text is not None
            estimate = pitch_tracker.update(samples, SAMPLE_RATE)
            if estimate.frequency_hz is not None:
                line.set_xdata([estimate.frequency_hz, estimate.frequency_hz])
                line.set_visible(True)
                text.set_text(_pitch_label(estimate))
            else:
                line.set_visible(False)
                text.set_text("")

        return image_holder

    interval_ms = max(1, int(1000 / TARGET_FPS))
    anim_ref[0] = FuncAnimation(
        fig,
        _update,
        interval=interval_ms,
        blit=True,
        cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        closed["flag"] = True
        _shutdown(anim_ref[0])


if __name__ == "__main__":
    run_live_bar_spectrum()
