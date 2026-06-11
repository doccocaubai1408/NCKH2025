# benchmark_worker.py
# ASCII-only.

import time
import psutil
from statistics import median

from ultralytics import YOLO

from config import (
    MODELS, IMG_SIZE, BENCH_SEC_PER_MODEL, BENCH_SLEEP_SEC, BENCH_TOPK,
    POWER_MAX_SAMPLES, BENCH_COOLDOWN_SEC
)
from db_benchmark import db_update_model
from quality_metrics import MistConfMeter
from learn_logger import LearnLogger


class BenchmarkWorker:
    def __init__(self, shared, db_ref, power_reader):
        self.shared = shared
        self.db = db_ref
        self.pwr = power_reader
        self.model_cache = {}
        self.last_bench_ts_by_ctx = {}

        self.process = psutil.Process()
        self.num_cores = psutil.cpu_count(logical=True)

        # 5 min learning logger
        self.learn_logger = LearnLogger(
            log_path="learn_log_5min.csv",
            summary_path="learn_summary_5min.txt",
            duration_sec=1800
        )

    def _clear_pending(self, ctx_id):
        try:
            if ctx_id is None:
                return
            with self.shared.bench_pending_lock:
                self.shared.bench_pending.discard(str(ctx_id))
        except Exception:
            pass

    def _bench_allowed(self):
        with self.shared.bench_ctrl_lock:
            return bool(self.shared.bench_enabled)

    def _db_is_in_ram(self):
        with self.shared.status_lock:
            return bool(getattr(self.shared, "db_in_ram", True))

    def _get_frame_copy(self):
        with self.shared.frame_lock:
            if self.shared.latest_frame is None:
                return None
            return self.shared.latest_frame.copy()

    def _benchmark_one_model(self, model, sec):
        meter = MistConfMeter()
        calls = 0
        I_samples = []

        t0 = time.time()
        while (time.time() - t0) < sec and (not self.shared.stop_flag):
            # hard gates
            if (not self._bench_allowed()) or (not self._db_is_in_ram()):
                break

            frame = self._get_frame_copy()
            if frame is None:
                time.sleep(0.01)
                continue

            if self.pwr is not None:
                I_A, pct, age = self.pwr.get_latest()
                if I_A is not None and len(I_samples) < POWER_MAX_SAMPLES:
                    I_samples.append(float(I_A))

            try:
                r = model(frame, imgsz=IMG_SIZE, verbose=False)
            except Exception:
                time.sleep(BENCH_SLEEP_SEC)
                continue

            calls += 1
            try:
                meter.update(frame, r[0], time.time())
            except Exception:
                pass

            time.sleep(BENCH_SLEEP_SEC)

        mist_rate, conf_rate = meter.get_rates()
        quality = 0.75 * float(mist_rate) + 0.25 * float(conf_rate)

        med_I = None
        if len(I_samples) >= 3:
            try:
                med_I = float(median(I_samples))
            except Exception:
                med_I = None
        elif len(I_samples) > 0:
            med_I = float(I_samples[len(I_samples) // 2])

        return quality, med_I, calls

    def _measure_model_bench(self, model, sec):
        cpu_before = self.process.cpu_times()
        t0 = time.perf_counter()

        quality, med_I, calls = self._benchmark_one_model(model, sec)

        t1 = time.perf_counter()
        cpu_after = self.process.cpu_times()

        duration = t1 - t0
        cpu_time = (
            cpu_after.user - cpu_before.user
            + cpu_after.system - cpu_before.system
        )

        cpu_total = (cpu_time / duration) * 100.0 if duration > 0 else 0.0
        cpu_machine = cpu_total / float(max(1, self.num_cores))

        return {
            "quality": quality,
            "current_A": med_I,
            "infer_calls": calls,
            "duration": duration,
            "cpu_total": cpu_total,
            "cpu_machine": cpu_machine
        }

    def run_forever(self):
        while not self.shared.stop_flag:
            self.learn_logger.save_summary_once()

            # If DB is unloaded (CRIT), do not benchmark at all.
            if (not self._bench_allowed()) or (not self._db_is_in_ram()):
                with self.shared.status_lock:
                    self.shared.bench_state = "DISABLED"
                time.sleep(0.2)
                continue

            try:
                job = self.shared.bench_queue.get(timeout=0.2)
            except Exception:
                continue

            ctx_id = job.get("context_id")
            ctx_obj = job.get("context_obj")

            # Re-check after dequeue (race-safe)
            if (not self._bench_allowed()) or (not self._db_is_in_ram()):
                with self.shared.status_lock:
                    self.shared.bench_state = "DISABLED"
                self._clear_pending(ctx_id)
                try:
                    self.shared.bench_queue.task_done()
                except Exception:
                    pass
                continue

            with self.shared.status_lock:
                self.shared.bench_state = "RUNNING"

            now = time.time()
            last_ts = self.last_bench_ts_by_ctx.get(ctx_id, 0.0)
            if (now - last_ts) < BENCH_COOLDOWN_SEC:
                with self.shared.status_lock:
                    self.shared.bench_state = "IDLE"
                self._clear_pending(ctx_id)
                try:
                    self.shared.bench_queue.task_done()
                except Exception:
                    pass
                continue
            self.last_bench_ts_by_ctx[ctx_id] = now

            model_list = MODELS[:]
            if BENCH_TOPK is not None:
                try:
                    k = int(BENCH_TOPK)
                    model_list = model_list[:max(1, k)]
                except Exception:
                    pass

            learn_t0 = time.perf_counter()
            learn_model_rows = []

            for (name, path) in model_list:
                if self.shared.stop_flag:
                    break
                if (not self._bench_allowed()) or (not self._db_is_in_ram()):
                    break

                if name not in self.model_cache:
                    try:
                        self.model_cache[name] = YOLO(path, task="detect")
                    except Exception as e:
                        print("Benchmark load failed:", name, str(e))
                        continue

                stat = self._measure_model_bench(
                    self.model_cache[name],
                    BENCH_SEC_PER_MODEL
                )

                quality = stat["quality"]
                med_I = stat["current_A"]
                calls = stat["infer_calls"]

                # One last hard gate before writing
                if (not self._bench_allowed()) or (not self._db_is_in_ram()):
                    break

                # Update ONLY on RAM
                with self.shared.db_lock:
                    db_update_model(
                        self.db,
                        ctx_id,
                        ctx_obj,
                        name,
                        quality,
                        med_I,
                        calls
                    )

                learn_model_rows.append({
                    "model": name,
                    "duration": stat["duration"],
                    "cpu_total": stat["cpu_total"],
                    "cpu_machine": stat["cpu_machine"],
                    "current_A": med_I,
                    "quality": quality,
                    "infer_calls": calls
                })

                # IMPORTANT: do NOT force db_in_ram=True here.
                with self.shared.status_lock:
                    # Only stamp updates when DB is actually in RAM
                    if bool(getattr(self.shared, "db_in_ram", True)):
                        self.shared.db_last_update_ts = time.time()
                        self.shared.db_last_update_reason = "BENCH"
                        self.shared.db_dirty = True
                        self.shared.db_state = "DIRTY(BENCH_RAM_ONLY)"

            learn_t1 = time.perf_counter()
            learn_duration = learn_t1 - learn_t0

            # One learning = all benchmarked models for one context.
            # This only logs during the first 5 minutes after program start.
            self.learn_logger.log_context(
                context_id=ctx_id,
                context_obj=ctx_obj,
                learn_duration_sec=learn_duration,
                model_rows=learn_model_rows
            )

            with self.shared.status_lock:
                self.shared.bench_state = "IDLE"

            self._clear_pending(ctx_id)

            try:
                self.shared.bench_queue.task_done()
            except Exception:
                pass

        try:
            self.learn_logger.close()
        except Exception:
            pass