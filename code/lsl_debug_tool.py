#!/usr/bin/env python3
"""
LSL Stream Debug Tool
Lists all available LSL streams and their properties
"""

import pylsl
import time

def list_lsl_streams():
    """List all available LSL streams"""
    print("🔍 Searching for LSL streams...")
    
    # Search for all streams
    streams = pylsl.resolve_streams(wait_time=5.0)
    
    if not streams:
        print("❌ No LSL streams found")
        print("\nTroubleshooting:")
        print("1. Make sure NIC-2 is running")
        print("2. Enable LSL in NIC-2: Protocol Settings → LSL Server → Enable")
        print("3. Check that Starstim device is connected")
        return
    
    print(f"✅ Found {len(streams)} LSL stream(s):\n")
    
    for i, stream in enumerate(streams):
        print(f"Stream {i+1}:")
        print(f"  Name: {stream.name()}")
        print(f"  Type: {stream.type()}")
        print(f"  Source ID: {stream.source_id()}")
        print(f"  Channels: {stream.channel_count()}")
        print(f"  Sample Rate: {stream.nominal_srate()} Hz")
        print(f"  Format: {stream.channel_format()}")
        print()
    
    # Look specifically for marker streams
    marker_streams = [s for s in streams if s.type() == 'Markers']
    if marker_streams:
        print(f"🎯 Found {len(marker_streams)} marker stream(s) - this is what we need!")
        test_marker_stream(marker_streams[0])
    else:
        print("⚠️  No marker streams found")
        print("   Looking for streams that might contain markers...")
        
        # Try to find streams that might have markers
        possible_marker_streams = [s for s in streams if 'marker' in s.name().lower() or 'event' in s.name().lower()]
        if possible_marker_streams:
            print(f"   Found {len(possible_marker_streams)} possible marker streams")
            test_marker_stream(possible_marker_streams[0])

def test_marker_stream(stream):
    """Test receiving data from a marker stream"""
    print(f"\n🧪 Testing marker stream: {stream.name()}")
    print("Listening for markers for 10 seconds...")
    print("(Start/stop stimulation in NIC-2 to generate markers)")
    
    inlet = pylsl.StreamInlet(stream)
    start_time = time.time()
    marker_count = 0
    
    while time.time() - start_time < 10.0:
        try:
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample:
                marker_count += 1
                print(f"  📍 Marker {marker_count}: {sample} (timestamp: {timestamp:.3f})")
        except:
            pass
    
    print(f"\n📊 Received {marker_count} markers in 10 seconds")
    if marker_count == 0:
        print("⚠️  No markers received - check NIC-2 LSL settings")
    else:
        print("✅ Marker stream is working!")

if __name__ == "__main__":
    list_lsl_streams()
