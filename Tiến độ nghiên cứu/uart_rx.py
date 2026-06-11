import serial

ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)

while True:
    data = ser.readline()
    if data:
        print("RAW:", data)
        print("TEXT:", data.decode("ascii", errors="replace"))