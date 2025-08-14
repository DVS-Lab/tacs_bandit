import pylsl
import time

def find_all_nic2_streams():
    """Find all LSL streams that might contain markers"""
    print("🔍 Scanning for ALL LSL streams...")
    
    # Look for any streams from NIC2
    all_streams = pylsl.resolve_streams(wait_time=3.0)
    
    print(f"📡 Found {len(all_streams)} total LSL streams:")
    
    nic2_streams = []
    for i, stream_info in enumerate(all_streams):
        name = stream_info.name()
        type_name = stream_info.type()
        channel_count = stream_info.channel_count()
        
        print(f"  {i+1}. Name: '{name}' | Type: '{type_name}' | Channels: {channel_count}")
        
        # Collect anything that might be from NIC2
        if any(keyword in name.lower() for keyword in ['nic', 'neuro', 'starstim', 'eeg', 'lsl']):
            nic2_streams.append((stream_info, name, type_name))
        elif name in ['', 'default'] or 'outlet' in name.lower():
            nic2_streams.append((stream_info, name, type_name))
    
    return nic2_streams

def test_stream_for_markers(stream_info, name, type_name, test_duration=20):
    """Test a specific stream for stimulation markers"""
    print(f"\n🧪 Testing stream: '{name}' (Type: {type_name})")
    
    try:
        inlet = pylsl.StreamInlet(stream_info)
        print(f"✅ Connected successfully!")
        print(f"⏱️  Listening for {test_duration} seconds...")
        
        # Special instructions for marker stream
        if 'marker' in name.lower() or type_name.lower() == 'markers':
            print("🎯 THIS IS THE MARKER STREAM!")
            print("👉 START your stimulation protocol in NIC2 NOW!")
            print("🔍 Looking for markers:")
            print("   201 = Ramp-up begins (perfect for task trigger!)")
            print("   203 = Full stimulation starts")
            print("   202 = Ramp-down begins")
            print("   204 = Stimulation stops")
        else:
            print("👉 You can start/stop stimulation, but markers likely in other streams")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < test_duration:
            try:
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample:
                    sample_count += 1
                    
                    # Check for stimulation markers
                    for value in sample:
                        if isinstance(value, (int, float)):
                            marker = int(value)
                            
                            # Detect specific stimulation markers
                            if marker == 201:
                                print(f"🚀 RAMP-UP BEGINS: Marker {marker} at {timestamp:.3f}")
                                print("   ⭐ PERFECT TRIGGER FOR BANDIT TASK!")
                            elif marker == 203:
                                print(f"⚡ FULL STIMULATION: Marker {marker} at {timestamp:.3f}")
                            elif marker == 202:
                                print(f"📉 RAMP-DOWN BEGINS: Marker {marker} at {timestamp:.3f}")
                            elif marker == 204:
                                print(f"⏹️  STIMULATION STOPS: Marker {marker} at {timestamp:.3f}")
                            elif 200 <= marker <= 210:
                                print(f"🎯 STIMULATION MARKER: {marker} at {timestamp:.3f}")
                            elif marker != 0 and abs(marker) > 10:  # Any significant non-zero marker
                                print(f"📍 Other marker: {marker} at {timestamp:.3f}")
                    
                    # Show sample format for first few samples
                    if sample_count <= 3:
                        print(f"📊 Sample {sample_count} format: {sample}")
                            
            except Exception as e:
                continue
                
        print(f"📈 Total samples received: {sample_count}")
        
        if sample_count == 0:
            print("❌ No data received from this stream")
        elif 'marker' in name.lower() and sample_count > 0:
            print("💡 This stream has data but may need stimulation to generate markers")
        
    except Exception as e:
        print(f"❌ Error connecting to stream: {e}")

def main():
    print("🚀 NIC2 Stimulation Marker Detective")
    print("🎯 Looking for ramp-up trigger (Marker 201)")
    print("=" * 50)
    
    # Find all possible streams
    streams = find_all_nic2_streams()
    
    if not streams:
        print("\n❌ No potential NIC2 streams found!")
        print("💡 Testing ALL streams anyway...")
        all_streams = pylsl.resolve_streams(wait_time=2.0)
        streams = [(s, s.name(), s.type()) for s in all_streams]
    
    print(f"\n🎯 Testing {len(streams)} streams for stimulation markers")
    print("\n💡 STRATEGY:")
    print("   • Let Quality/Accelerometer streams run without testing")
    print("   • START stimulation when testing 'Markers' stream")
    print("   • Also test EEG stream for embedded markers")
    
    # Test each stream
    for i, (stream_info, name, type_name) in enumerate(streams):
        
        # Skip non-essential streams to focus on markers
        if type_name.lower() in ['quality', 'accelerometer'] and len(streams) > 2:
            print(f"\n⏭️  Skipping '{name}' - focusing on marker streams")
            continue
            
        if i > 0:
            input(f"\nPress Enter to test next stream...")
            
        test_stream_for_markers(stream_info, name, type_name)
    
    print("\n" + "="*50)
    print("✅ Detection complete!")
    print("\n🎯 NEXT STEPS:")
    print("1. If marker 201 was detected → Use that stream for task triggering")
    print("2. If no markers found → Check NIC2 settings or device connection")
    print("3. Implement final bandit task with the working stream")

if __name__ == "__main__":
    main()
