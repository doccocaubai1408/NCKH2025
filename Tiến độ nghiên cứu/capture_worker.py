# capture_worker.py
# ASCII-only.

import time
import cv2
from config import CAM_ID

class CaptureWorker:
    def __init__(self, shared):
        self.shared = shared

    def run_forever(self):
        cap = cv2.VideoCapture(CAM_ID)
        if not cap.isOpened():
            print("Cannot open camera")
            self.shared.stop_flag = True
            return

        while not self.shared.stop_flag:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ts = time.time()
            with self.shared.frame_lock:
                self.shared.latest_frame = frame
                self.shared.latest_ts = ts

        cap.release()
