import pylsl
import time

print("🔍 Comprehensive Marker Detection Test")
print("=" * 45)

# Find all streams
streams = pylsl.resolve_streams(wait_time=5.0)
print(f"📡 Found {len(streams)} streams:")
for i, stream in enumerate(streams):
    print(f"  {i+1}. {stream.name()} ({stream.type()})")

# Connect to marker stream
marker_stream = None
for stream in streams:
    if 'marker' in stream.name().lower():
        marker_stream = stream
        break

if marker_stream:
    inlet = pylsl.StreamInlet(marker_stream)
    print(f"\n✅ Connected to: {marker_stream.name()}")
    
    print("\n🧪 TEST 1: Check if stream is sending ANY data")
    print("⏱️ Listening for 10 seconds...")
    
    start_time = time.time()
    sample_count = 0
    all_markers = []
    
    while time.time() - start_time < 10:
        try:
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if sample:
                sample_count += 1
                marker = sample[0]
                all_markers.append(marker)
                
                if sample_count <= 5:  # Show first few samples
                    print(f"📊 Sample {sample_count}: {marker} at {timestamp:.3f}")
                    
        except:
            continue
    
    print(f"\n📈 Received {sample_count} samples in 10 seconds")
    if all_markers:
        unique_markers = set(all_markers)
        print(f"🎯 Unique markers seen: {sorted(unique_markers)}")
    
    if sample_count == 0:
        print("❌ No data flowing - this might be the issue!")
        print("\n💡 TROUBLESHOOTING:")
        print("1. Is a Starstim device connected to NIC2?")
        print("2. Is a protocol loaded in NIC2?")
        print("3. Are you running NIC2 in the right mode?")
    else:
        print(f"\n✅ Stream is active with {sample_count} samples")
        print("\n🧪 TEST 2: Looking for stimulation markers during protocol")
        print("👉 START your stimulation protocol in NIC2 NOW!")
        print("⏱️ Listening for 30 seconds...")
        
        start_time = time.time()
        stim_markers = []
        
        while time.time() - start_time < 30:
            try:
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample:
                    marker = int(sample[0])
                    
                    # Log ALL non-zero markers
                    if marker != 0:
                        stim_markers.append(marker)
                        print(f"📍 Marker detected: {marker} at {timestamp:.3f}")
                        
                        # Check for known stimulation markers
                        if marker == 201:
                            print("🚀 RAMP-UP BEGINS!")
                        elif marker == 203:
                            print("⚡ FULL STIMULATION!")
                        elif marker == 202:
                            print("📉 RAMP-DOWN!")
                        elif marker == 204:
                            print("⏹️ STIMULATION STOPS!")
                        elif 200 <= marker <= 210:
                            print("🎯 Unknown stimulation marker!")
                            
            except:
                continue
        
        print(f"\n📊 Stimulation markers detected: {stim_markers}")
        
        if not stim_markers:
            print("\n❌ No stimulation markers detected during protocol run")
            print("\n💡 POSSIBLE ISSUES:")
            print("1. Markers might only send when device is physically connected")
            print("2. Stimulation might need to be actively running")
            print("3. NIC2 settings might need adjustment")
            print("4. This might be a simulation/offline mode")

else:
    print("❌ No marker stream found!")
