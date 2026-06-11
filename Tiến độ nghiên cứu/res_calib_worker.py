# res_calib_worker.py
# ASCII-only.
#
# "Co-driver" calibration worker.
# IMPORTANT:
# - No cv2.imshow here (OpenCV GUI is not thread-safe; can freeze).
# - Use snapshot frame passed by realtime to calibrate (stable + cheap).
#
# Result: updates shared.res_lockw_by_light[light] and shared.res_active_w.

import time

from ultralytics import YOLO

from config import MODELS
from adaptive_res import auto_find_min_good_width_on_frame

# Calib params (keep aligned with realtime_worker.py defaults)
CONF_TH = 0.25
TEST_FRAMES = 10
MIN_DET_RATE = 0.30
SHRINK_RATIO = 0.85
MIN_W = 96
BINARY_ITERS = 10

class ResCalibWorker:
    def __init__(self, shared):
        self.shared = shared
        self.model_cache = {}

    def _load_model(self, model_name):
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        path_map = {n: p for (n, p) in MODELS}
        path = path_map.get(model_name, MODELS[0][1])
        m = YOLO(path, task="detect")
        self.model_cache[model_name] = m
        return m

    def run_forever(self):
        while not self.shared.stop_flag:
            try:
                job = self.shared.res_queue.get(timeout=0.2)
            except Exception:
                continue

            model_name = job.get("model_name")
            light_name = job.get("light_name")
            frame = job.get("frame_bgr")
            base_w = job.get("base_w")

            if frame is None or base_w is None or model_name is None or light_name is None:
                try:
                    self.shared.res_queue.task_done()
                except Exception:
                    pass
                continue

            with self.shared.res_lock:
                self.shared.res_running = True
                self.shared.res_last_msg = "CALIB_START"
                self.shared.res_active_light = str(light_name)

            try:
                model = self._load_model(model_name)

                # Non-visual auto limit on a snapshot frame.
                w = auto_find_min_good_width_on_frame(
                    model=model,
                    frame_bgr=frame,
                    base_w=int(base_w),
                    shrink_ratio=SHRINK_RATIO,
                    min_w=MIN_W,
                    test_frames=TEST_FRAMES,
                    min_det_rate=MIN_DET_RATE,
                    binary_iters=BINARY_ITERS,
                    conf_th=CONF_TH,
                )

                with self.shared.res_lock:
                    if str(light_name) in self.shared.res_lockw_by_light:
                        self.shared.res_lockw_by_light[str(light_name)] = int(w)
                    self.shared.res_active_w = int(w)
                    self.shared.res_last_msg = "CALIB_DONE w=%d" % int(w)

            except Exception as e:
                with self.shared.res_lock:
                    self.shared.res_last_msg = "CALIB_ERR %s" % str(e)[:120]
            finally:
                with self.shared.res_lock:
                    self.shared.res_running = False
                try:
                    self.shared.res_queue.task_done()
                except Exception:
                    pass

            time.sleep(0.01)
