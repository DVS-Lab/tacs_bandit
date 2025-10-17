#!/usr/bin/env python3
"""
LSL Stream Detection Test
Run this on the desktop to detect LSL streams from the laptop/NIC-2
"""

import pylsl
import time

def test_lsl_connection():
    """Test for LSL streams from NIC-2/Starstim"""
    
    print("Searching for LSL streams...")
    print("Make sure NIC-2 is running with LSL enabled on the laptop")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Look for streams with 5-second timeout
            streams = pylsl.resolve_streams(wait_time=5.0)
            
            if streams:
                print(f"Found {len(streams)} LSL stream(s):")
                for i, stream in enumerate(streams):
                    print(f"  Stream {i+1}:")
                    print(f"    Name: {stream.name()}")
                    print(f"    Type: {stream.type()}")
                    print(f"    Source ID: {stream.source_id()}")
                    print(f"    Channels: {stream.channel_count()}")
                    print(f"    Sample Rate: {stream.nominal_srate()}")
                    print(f"    Format: {stream.channel_format()}")
                    print()
                
                # Try to connect to the first marker stream
                marker_streams = [s for s in streams if s.type() == 'Markers']
                if marker_streams:
                    print("Attempting to connect to marker stream...")
                    inlet = pylsl.StreamInlet(marker_streams[0])
                    
                    print("Listening for markers (10 seconds)...")
                    start_time = time.time()
                    while time.time() - start_time < 10:
                        sample, timestamp = inlet.pull_sample(timeout=1.0)
                        if sample:
                            print(f"Received marker: {sample[0]} at time {timestamp}")
                    
                    print("Test complete!")
                    break
                else:
                    print("No marker streams found. Make sure LSL is enabled in NIC-2.")
            else:
                print("No LSL streams detected. Retrying...")
                
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_lsl_connection()
