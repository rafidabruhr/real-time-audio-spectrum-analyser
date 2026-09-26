import argparse
import sys
import time

import numpy as np

from audio_capture import AudioCaptureError, AudioStream, AudioStreamReadError, list_devices, list_loopback_devices
from config import CHUNK_SIZE, DEFAULT_SOURCE, SAMPLE_RATE, WINDOW_TYPE
from visualizer import run_live_bar_spectrum, run_live_waterfall, run_static_bar_spectrum


def _chunk_stats(chunk):
    x = chunk.astype(np.float64)
    return float(x.min()), float(x.max()), float(np.abs(x).mean())


def run_capture_test(duration_sec=5.0, source="mic", device=None):
    print(f"Sample rate: {SAMPLE_RATE} Hz, chunk size: {CHUNK_SIZE} samples", flush=True)
    print(f"Capturing ({source}) for {duration_sec:.1f} s — "
          f"{'speak or clap near the mic' if source == 'mic' else 'play some audio'}.\n", flush=True)

    try:
        with AudioStream(source=source, device=device) as stream:
            print(f"Backend: {stream.backend}\n")
            deadline = time.monotonic() + duration_sec
            i = 0
            while time.monotonic() < deadline:
                try:
                    chunk = stream.read_chunk()
                except AudioStreamReadError as err:
                    print(err, file=sys.stderr)
                    sys.exit(1)
                lo, hi, mean_abs = _chunk_stats(chunk)
                print(f"chunk {i:4d}  min={lo:+.5f}  max={hi:+.5f}  mean|x|={mean_abs:.5f}")
                i += 1
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)

    print("\nDone. In silence, mean|x| should stay near zero; speech/claps/playback should raise max and mean|x|.")


def _print_device_list():
    try:
        devices = list_devices()
    except AudioCaptureError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    print("All devices:")
    for d in devices:
        print(f"  [{d['index']:2d}] {d['name']}  (in={d['max_input_channels']}, "
              f"out={d['max_output_channels']}, hostapi={d['hostapi']})")

    loopback = list_loopback_devices()
    print("\nLoopback-capable (for --source loopback --device N):")
    if loopback:
        for d in loopback:
            print(f"  [{d['index']:2d}] {d['name']}")
    else:
        print("  (none detected)\n"
              "  Windows: any WASAPI output device works automatically.\n"
              "  Linux: look for a PulseAudio/PipeWire 'Monitor of ...' input.\n"
              "  macOS: install a virtual device such as BlackHole and re-run.")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time microphone/loopback spectrum analyzer (FFT + waterfall + pitch).",
        epilog="FFT windows: 'hann' (default) balances leakage and resolution; "
               "'hamming' similar; 'rectangular' shows maximum leakage for lab comparison.",
    )
    parser.add_argument("--capture-test", action="store_true", help="Phase 1: print min/max/mean per chunk for 5 seconds.")
    parser.add_argument("--static-bars", action="store_true", help="Phase 2: capture one frame and show a static dB bar spectrum.")
    parser.add_argument("--mode", default="bars", choices=("bars", "waterfall"), help="Live view: bar spectrum or scrolling waterfall spectrogram.")
    parser.add_argument("--live-bars", action="store_true", help="Shortcut for --mode bars.")
    parser.add_argument("--window", default=WINDOW_TYPE, choices=("hann", "hamming", "rectangular"), help="FFT window function.")
    parser.add_argument("--pitch", action="store_true", help="Overlay detected pitch (note name + Hz) on the bar/waterfall views.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, choices=("mic", "loopback"), help="Capture source: microphone, or system loopback.")
    parser.add_argument("--device", type=int, default=None, help="Explicit device index (see --list-devices). Required for loopback on Linux/macOS.")
    parser.add_argument("--list-devices", action="store_true", help="Print available audio devices (and loopback candidates) and exit.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_devices:
        _print_device_list()
        sys.exit(0)

    if args.static_bars:
        run_static_bar_spectrum(window_type=args.window, show_pitch=args.pitch, source=args.source, device=args.device)
    elif args.capture_test:
        run_capture_test(source=args.source, device=args.device)
    else:
        mode = "bars" if args.live_bars else args.mode
        if mode == "waterfall":
            run_live_waterfall(window_type=args.window, show_pitch=args.pitch, source=args.source, device=args.device)
        else:
            run_live_bar_spectrum(window_type=args.window, show_pitch=args.pitch, source=args.source, device=args.device)