"""
Entry point for the spectrum analyzer project.

Phase 1: ``--capture-test`` — print amplitude stats for 5 seconds.
Phase 2: ``--static-bars`` — one-shot frequency bar chart.
Live analyzer: ``python main.py [--mode bars|waterfall] [--window TYPE]``.
Extension: ``--pitch`` overlays detected fundamental frequency (note + Hz);
``--source loopback`` captures system audio instead of the mic; ``--device``
picks a specific device (see ``--list-devices``).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from audio_capture import (
    AudioCaptureError,
    AudioStream,
    AudioStreamReadError,
    list_devices,
    list_loopback_devices,
)
from config import CHUNK_SIZE, DEFAULT_SOURCE, SAMPLE_RATE, WINDOW_TYPE
from visualizer import (
    run_live_bar_spectrum,
    run_live_waterfall,
    run_static_bar_spectrum,
)


def _chunk_stats(chunk: np.ndarray) -> tuple[float, float, float]:
    """Min, max, and mean absolute amplitude for a single buffer."""
    x = chunk.astype(np.float64, copy=False)
    return float(np.min(x)), float(np.max(x)), float(np.mean(np.abs(x)))


def run_capture_test(
    duration_sec: float = 5.0,
    source: str = "mic",
    device: "int | None" = None,
) -> None:
    """
    Open the capture stream, read chunks for ``duration_sec`` seconds, print stats.

    Parameters
    ----------
    duration_sec : float
        How long to capture before exiting.
    source : "mic" | "loopback"
        Capture source.
    device : int, optional
        Explicit device index.
    """
    print(f"Sample rate: {SAMPLE_RATE} Hz, chunk size: {CHUNK_SIZE} samples", flush=True)
    print(
        f"Capturing ({source}) for {duration_sec:.1f} s — "
        f"{'speak or clap near the mic' if source == 'mic' else 'play some audio'}.\n",
        flush=True,
    )

    try:
        with AudioStream(source=source, device=device) as stream:
            print(f"Backend: {stream.backend}\n")
            deadline = time.monotonic() + duration_sec
            chunk_index = 0
            while time.monotonic() < deadline:
                try:
                    chunk = stream.read_chunk()
                except AudioStreamReadError as err:
                    print(err, file=sys.stderr)
                    sys.exit(1)
                lo, hi, mean_abs = _chunk_stats(chunk)
                print(
                    f"chunk {chunk_index:4d}  "
                    f"min={lo:+.5f}  max={hi:+.5f}  mean|x|={mean_abs:.5f}"
                )
                chunk_index += 1
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)

    print("\nDone. In silence, mean|x| should stay near zero; "
          "speech/claps/playback should raise max and mean|x|.")


def _print_device_list() -> None:
    """Print all audio devices plus which ones look loopback-capable."""
    try:
        devices = list_devices()
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    print("All devices:")
    for d in devices:
        print(
            f"  [{d['index']:2d}] {d['name']}  "
            f"(in={d['max_input_channels']}, out={d['max_output_channels']}, "
            f"hostapi={d['hostapi']})"
        )

    loopback = list_loopback_devices()
    print("\nLoopback-capable (for --source loopback --device N):")
    if loopback:
        for d in loopback:
            print(f"  [{d['index']:2d}] {d['name']}")
    else:
        print(
            "  (none detected)\n"
            "  Windows: any WASAPI output device works automatically.\n"
            "  Linux: look for a PulseAudio/PipeWire 'Monitor of ...' input.\n"
            "  macOS: install a virtual device such as BlackHole and re-run."
        )


def _build_parser() -> argparse.ArgumentParser:
    # Window choice (spectral leakage):
    # • rectangular — narrowest main lobe if tone sits on a bin, but strong
    #   side lobes; bright "skirts" on the waterfall when pitch is between bins.
    # • hann / hamming — wider main lobe (peak smears across more bins) but
    #   much lower side lobes; cleaner visuals and less audible leakage between
    #   partials when analyzing voice or music.
    parser = argparse.ArgumentParser(
        description="Real-time microphone/loopback spectrum analyzer (FFT + waterfall + pitch).",
        epilog=(
            "FFT windows: 'hann' (default) balances leakage and resolution; "
            "'hamming' similar; 'rectangular' shows maximum leakage for lab "
            "comparison."
        ),
    )
    parser.add_argument(
        "--capture-test",
        action="store_true",
        help="Phase 1: print min/max/mean per chunk for 5 seconds.",
    )
    parser.add_argument(
        "--static-bars",
        action="store_true",
        help="Phase 2: capture one frame and show a static dB bar spectrum.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bars",
        choices=("bars", "waterfall"),
        help="Live view: bar spectrum or scrolling waterfall spectrogram.",
    )
    parser.add_argument(
        "--live-bars",
        action="store_true",
        help="Shortcut for --mode bars.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=WINDOW_TYPE,
        choices=("hann", "hamming", "rectangular"),
        help="FFT window function.",
    )
    parser.add_argument(
        "--pitch",
        action="store_true",
        help="Overlay detected pitch (note name + Hz) on the bar/waterfall views.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        choices=("mic", "loopback"),
        help="Capture source: microphone, or system loopback ('what you hear').",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Explicit device index (see --list-devices). Required for "
             "loopback on Linux/macOS.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices (and loopback candidates) and exit.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_devices:
        _print_device_list()
        sys.exit(0)

    if args.static_bars:
        run_static_bar_spectrum(
            window_type=args.window,
            show_pitch=args.pitch,
            source=args.source,
            device=args.device,
        )
    elif args.capture_test:
        run_capture_test(source=args.source, device=args.device)
    else:
        mode = "bars" if args.live_bars else args.mode
        if mode == "waterfall":
            run_live_waterfall(
                window_type=args.window,
                show_pitch=args.pitch,
                source=args.source,
                device=args.device,
            )
        else:
            run_live_bar_spectrum(
                window_type=args.window,
                show_pitch=args.pitch,
                source=args.source,
                device=args.device,
            )
