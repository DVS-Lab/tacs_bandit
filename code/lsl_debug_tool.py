#!/usr/bin/env python3
"""
LSL stream inspection helper for NIC-2 markers and StarStim EEG.
"""

from __future__ import annotations

import time

import pylsl


def describe_stream(stream):
    print(f"  Name: {stream.name()}")
    print(f"  Type: {stream.type()}")
    print(f"  Source ID: {stream.source_id()}")
    print(f"  Channels: {stream.channel_count()}")
    print(f"  Sample Rate: {stream.nominal_srate()} Hz")
    print(f"  Format: {stream.channel_format()}")
    print()


def list_lsl_streams():
    print("Searching for available LSL streams...\n")
    streams = pylsl.resolve_streams(wait_time=5.0)

    if not streams:
        print("No LSL streams found.")
        print("\nTroubleshooting:")
        print("1. Make sure NIC-2 is running.")
        print("2. Enable LSL in NIC-2: Protocol Settings -> LSL Server -> Enable.")
        print("3. Check that StarStim is connected and sharing EEG over LSL.")
        return

    print(f"Found {len(streams)} stream(s):\n")
    for index, stream in enumerate(streams, start=1):
        print(f"Stream {index}:")
        describe_stream(stream)

    marker_streams = [stream for stream in streams if stream.type() == "Markers"]
    eeg_streams = [stream for stream in streams if stream.type().upper() == "EEG"]

    if marker_streams:
        print(f"Marker streams detected: {len(marker_streams)}")
        test_marker_stream(marker_streams[0])
    else:
        print("No marker streams detected.\n")

    if eeg_streams:
        print(f"\nEEG streams detected: {len(eeg_streams)}")
        test_eeg_stream(eeg_streams[0])
    else:
        print("No EEG streams detected.")


def test_marker_stream(stream):
    print(f"\nTesting marker stream: {stream.name()}")
    print("Listening for markers for 15 seconds...")
    inlet = pylsl.StreamInlet(stream)
    marker_count = 0
    start_time = time.time()
    while time.time() - start_time < 15.0:
        sample, timestamp = inlet.pull_sample(timeout=0.1)
        if sample:
            marker_count += 1
            print(f"  Marker {marker_count}: {sample} at {timestamp:.3f}")
    if marker_count == 0:
        print("  No markers received. Start or stop a NIC-2 protocol to confirm marker flow.")


def test_eeg_stream(stream):
    print(f"\nTesting EEG stream: {stream.name()}")
    print("Listening for EEG samples for 5 seconds...")
    inlet = pylsl.StreamInlet(stream)
    sample_count = 0
    start_time = time.time()
    while time.time() - start_time < 5.0:
        chunk, timestamps = inlet.pull_chunk(timeout=0.2, max_samples=128)
        if chunk:
            sample_count += len(chunk)
    print(f"  Received {sample_count} EEG samples in 5 seconds.")


if __name__ == "__main__":
    list_lsl_streams()
