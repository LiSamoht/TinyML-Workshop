import serial
import time

# Configuration - CHANGED TO COM5
PORT = 'COM5'  # Your Arduino is here!
BAUDRATE = 9600
OUTPUT_FILE = 'sensorlog.csv'

print(f"Opening {PORT}...")

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Wait for connection
    
    print(f"Successfully connected to {PORT}")
    print(f"Recording data to {OUTPUT_FILE}")
    print("Press Ctrl+C to stop\n")
    
    with open(OUTPUT_FILE, 'w') as f:
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(line)
                        f.write(line + '\n')
                        f.flush()
            except KeyboardInterrupt:
                print("\nStopped.")
                break
                
except serial.SerialException as e:
    print(f"\nError: {e}")
    print("\nMake sure Arduino IDE is closed!")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()