# shared_state.py
# ASCII-only.

import threading
import queue
import time


class SharedState:
    def __init__(self):
        self.stop_flag = False

        # latest frame
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0.0

        # db lock + data loaded elsewhere
        self.db_lock = threading.Lock()

        # model lock (optional)
        self.model_lock = threading.Lock()

        # benchmark job queue
        self.bench_queue = queue.Queue(maxsize=4)

        # NEW: contexts that are already queued/running benchmark
        self.bench_pending_lock = threading.Lock()
        self.bench_pending = set()

        # rate monitor (realtime loop time)
        self.rt_lock = threading.Lock()
        self.rt_last_ms = 0.0

        # ===== benchmark control gate =====
        self.bench_ctrl_lock = threading.Lock()
        self.bench_enabled = True

        # ===== system status for realtime overlay =====
        self.status_lock = threading.Lock()
        self.sys_mode = "NORMAL"      # NORMAL / CRIT
        self.bench_state = "IDLE"     # IDLE / RUNNING / DISABLED
        self.db_state = "LOADED"      # LOADED / SAVED / RELOADED / DIRTY(...)

        # ===== DB RAM status tracking (NEW) =====
        # db_in_ram: True if contexts are present in RAM (not unloaded).
        self.db_in_ram = True
        self.db_loaded_ts = time.time()
        self.db_last_update_ts = 0.0
        self.db_last_update_reason = "NONE"   # BENCH / ONLINE / RELOAD / SAVE / NONE
        self.db_dirty = False
        self.db_last_save_ts = 0.0

        # ===== RES (adaptive resolution) shared state =====
        self.res_lock = threading.Lock()
        self.res_lockw_by_light = {
            "BRIGHT": None,
            "NORMAL": None,
            "DIM": None,
            "DARK": None,
        }
        self.res_running = False
        self.res_last_msg = ""
        self.res_active_light = None
        self.res_active_w = None

        # jobs for calibration worker (non-blocking)
        self.res_queue = queue.Queue(maxsize=2)