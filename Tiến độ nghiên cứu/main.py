# main.py
# ASCII-only.

import threading
import time

from shared_state import SharedState
from db_benchmark import db_load, db_save_atomic
from power_serial import SerialPowerReader
from capture_worker import CaptureWorker
from benchmark_worker import BenchmarkWorker
from realtime_worker import RealtimeWorker
from res_calib_worker import ResCalibWorker


def main():
    shared = SharedState()

    # load db
    db = db_load()

    # record DB loaded
    with shared.status_lock:
        shared.db_in_ram = True
        shared.db_loaded_ts = time.time()
        shared.db_state = "LOADED"

    # power
    pwr = SerialPowerReader()
    pwr.start()
    if not pwr.ok:
        pwr = None

    capw = CaptureWorker(shared)
    benw = BenchmarkWorker(shared, db, pwr)
    rt_w = RealtimeWorker(shared, db, pwr)
    resw = ResCalibWorker(shared)

    th_cap = threading.Thread(target=capw.run_forever, daemon=True)
    th_ben = threading.Thread(target=benw.run_forever, daemon=True)
    th_rt = threading.Thread(target=rt_w.run_forever, daemon=True)
    th_res = threading.Thread(target=resw.run_forever, daemon=True)

    th_cap.start()
    th_ben.start()
    th_rt.start()
    th_res.start()

    try:
        while not shared.stop_flag:
            time.sleep(0.2)
    finally:
        shared.stop_flag = True
        try:
            if pwr is not None:
                pwr.close()
        except Exception:
            pass

        # If DB is UNLOADED, do NOT save empty DB over disk.
        with shared.status_lock:
            in_ram = bool(getattr(shared, "db_in_ram", True))

        ok = True
        did_save = False

        if in_ram:
            with shared.db_lock:
                ok = db_save_atomic(db)
            did_save = True

        with shared.status_lock:
            shared.db_last_save_ts = time.time()
            if did_save:
                shared.db_dirty = False
                shared.db_state = "SAVED(EXIT)" if ok else "SAVE_FAIL(EXIT)"
            else:
                shared.db_state = "SKIP_SAVE(EXIT_UNLOADED)"

        print("Exit.")


if __name__ == "__main__":
    main()