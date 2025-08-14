import pylsl
import time

print("🎯 Testing for Marker 201 (Ramp-up Begins)")
print("=" * 40)

# Connect to the marker stream
streams = pylsl.resolve_streams(wait_time=5.0)
if streams:
    inlet = pylsl.StreamInlet(streams[0])
    print("✅ Connected to NIC2_Streaming-Markers")
    print("\n⏳ Ready to detect markers...")
    print("👉 WAIT - then START a NEW stimulation protocol in NIC2")
    print("🔍 Looking for marker 201 (ramp-up begins)...")
    
    for i in range(60):  # 60 seconds to catch it
        try:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                marker = int(sample[0])
                
                if marker == 201:
                    print(f"🚀 PERFECT! Ramp-up begins: {marker} at {timestamp:.3f}")
                    print("⭐ This is your bandit task trigger!")
                    break
                elif marker == 203:
                    print(f"⚡ Full stimulation: {marker} at {timestamp:.3f}")
                elif marker == 202:
                    print(f"📉 Ramp-down: {marker} at {timestamp:.3f}")
                elif marker == 204:
                    print(f"⏹️  Stimulation stops: {marker} at {timestamp:.3f}")
                else:
                    print(f"📍 Other marker: {marker} at {timestamp:.3f}")
                    
        except:
            continue
            
    print("✅ Test complete!")
else:
    print("❌ Could not find marker stream")
