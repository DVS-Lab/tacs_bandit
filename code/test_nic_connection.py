import socket
import time

def test_nic_communication():
    host = '192.168.1.68'  # Your Mac's IP
    port = 1234
    
    try:
        print(f"Connecting to NIC-2 at {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        print("✅ Connected successfully!")
        
        # Try to receive any initial data from NIC-2
        sock.settimeout(2.0)
        try:
            initial_data = sock.recv(1024)
            print(f"📥 NIC-2 sent: {initial_data[:100]}...")  # First 100 bytes
        except socket.timeout:
            print("📭 No initial data from NIC-2")
        
        # Try sending some test commands
        test_commands = [
            "STATUS\n",
            "GET_STATUS\n", 
            "HELLO\n",
            "VERSION\n"
        ]
        
        for cmd in test_commands:
            print(f"📤 Sending: {cmd.strip()}")
            sock.send(cmd.encode())
            
            try:
                response = sock.recv(1024)
                print(f"📥 Response: {response}")
            except socket.timeout:
                print("📭 No response")
            
            time.sleep(0.5)
        
        sock.close()
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_nic_communication()
