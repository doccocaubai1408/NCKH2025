# power_serial.py
# ASCII-only.

import time
import threading

try:
    import serial
except Exception:
    serial = None

from config import SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT, SERIAL_MAX_AGE_SEC

class SerialPowerReader:
    def __init__(self):
        self.ok = False
        self.lock = threading.Lock()
        self.last_ts = 0.0
        self.last_I = None
        self.last_pct = None

        self.stop = False
        self.th = None
        self.ser = None

    def start(self):
        if serial is None:
            return
        try:
            self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
            self.ok = True
        except Exception:
            self.ok = False
            return
        self.stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self.stop:
            try:
                line = self.ser.readline()
            except Exception:
                time.sleep(0.01)
                continue
            if not line:
                continue
            try:
                s = line.decode("utf-8", errors="ignore").strip()
                parts = s.split(",")
                if len(parts) < 2:
                    continue
                I_A = float(parts[0])
                pct = float(parts[1])
                ts = time.time()
                with self.lock:
                    self.last_ts = ts
                    self.last_I = I_A
                    self.last_pct = pct
            except Exception:
                continue

    def get_latest(self):
        if not self.ok:
            return None, None, 1e9
        with self.lock:
            ts = self.last_ts
            I_A = self.last_I
            pct = self.last_pct
        if ts <= 0.0 or I_A is None or pct is None:
            return None, None, 1e9
        age = time.time() - ts
        if age > SERIAL_MAX_AGE_SEC:
            return None, None, age
        return float(I_A), float(pct), age

    def close(self):
        self.stop = True
        if self.th is not None:
            try:
                self.th.join(timeout=0.5)
            except Exception:
                pass
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
