# realtime_worker.py
# ASCII-only.

import time
import cv2
from statistics import median

from ultralytics import YOLO

from config import MODELS, IMG_SIZE, CONTEXT_STABLE_SEC, MIN_SWITCH_SEC
from context_features import ContextTracker
from policy import nearest_context_id, choose_model_from_context, weights_for_pin
from db_benchmark import db_save_atomic, db_load, db_update_model, db_default

from quality_metrics import MistConfMeter
from adaptive_res import resize_keep_aspect, make_canvas, auto_find_min_good_width_visual_shared
from power_mode import compute_global_model_stats, pick_global_model_crit


CRIT_THRESHOLD = 20.0
EXIT_CRIT_THRESHOLD = 40.0

ONLINE_ENABLE = True
ONLINE_WIN_SEC = 4.0
ONLINE_MIN_CALLS = 8
ONLINE_RANK_CUTOFF = 3
ONLINE_HYST_REL = 0.05
ONLINE_UPDATE_COOLDOWN_SEC = 8.0
ONLINE_MAX_I_SAMPLES = 400

RES_ENABLE_ON_CRIT = True
RES_SHRINK_RATIO = 0.85
RES_MIN_W = 96
RES_GOOD_STREAK_N = 2
RES_MISS_STREAK_N = 2
LAST_OBJ_MAX_AGE_SEC = 20.0
RES_MIN_RETEST_SEC = 2.0

CONF_TH = 0.25
TEST_FRAMES = 40
MIN_DET_RATE = 0.30
SHRINK_RATIO = 0.85
MIN_W = 96
BINARY_ITERS = 10
MIN_RETEST_SEC = 3.0

# Unload DB for real on CRIT (after saving)
DB_UNLOAD_ON_CRIT = True


def _reload_db_inplace(dst_db, src_db):
    dst_db.clear()
    for k, v in src_db.items():
        dst_db[k] = v


def _fmt_hms(ts):
    if ts is None or float(ts) <= 0.0:
        return "NA"
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return "NA"


class RealtimeWorker:
    def __init__(self, shared, db_ref, power_reader):
        self.shared = shared
        self.db = db_ref
        self.pwr = power_reader

        self.ctx_tracker = ContextTracker()

        self.model_cache = {}
        self.current_name = None
        self.current_model = None
        self.last_switch_ts = 0.0

        self.last_ctx_seen = None
        self.ctx_since = None

        self.in_crit = False

        self.base_w = None
        self.base_h = None

        self.lock_w = None
        self.calibrated = False
        self.last_retest_ts = 0.0

        self.last_obj_frame = None
        self.last_obj_ts = 0.0

        # resize cache per light (RAM-only)
        self.res_cache = {}
        self.res_search_active = False
        self.res_search_light = None
        self.res_search_src = "LIVE"
        self.res_trial_w = None
        self.res_last_good_w = None
        self.res_good_streak = 0
        self.res_miss_streak = 0
        self.res_last_start_ts = 0.0

        # online window
        self.on_win_t0 = 0.0
        self.on_use_cid = None
        self.on_pin_bin = None
        self.on_model_name = None
        self.on_meter = MistConfMeter()
        self.on_calls = 0
        self.on_I_samples = []
        self.on_last_update_ts = 0.0

    def _get_frame_copy(self):
        with self.shared.frame_lock:
            if self.shared.latest_frame is None:
                return None
            return self.shared.latest_frame.copy()

    def _load_model(self, name):
        if name in self.model_cache:
            return self.model_cache[name]
        path_map = {n: p for (n, p) in MODELS}
        path = path_map.get(name, MODELS[0][1])
        m = YOLO(path, task="detect")
        self.model_cache[name] = m
        return m

    def _ensure_initial_model(self):
        if self.current_model is None:
            self.current_name = MODELS[0][0]
            self.current_model = self._load_model(self.current_name)
            self.last_switch_ts = time.time()

    def _infer(self, frame_bgr):
        try:
            r = self.current_model(frame_bgr, imgsz=IMG_SIZE, verbose=False)
            out = r[0].plot()
            return out, r
        except Exception:
            return frame_bgr, None

    def _has_obj(self, results, conf_th):
        if results is None:
            return False, 0.0, 0
        try:
            r0 = results[0]
            if r0.boxes is None or len(r0.boxes) == 0:
                return False, 0.0, 0
            confs = []
            for b in r0.boxes:
                c = float(b.conf)
                if c >= float(conf_th):
                    confs.append(c)
            if not confs:
                return False, 0.0, 0
            return True, float(max(confs)), int(len(confs))
        except Exception:
            return False, 0.0, 0

    # ---------- ONLINE ----------
    def _online_reset(self, use_cid, pin_bin, model_name):
        self.on_win_t0 = time.time()
        self.on_use_cid = str(use_cid)
        self.on_pin_bin = str(pin_bin)
        self.on_model_name = str(model_name)
        self.on_meter = MistConfMeter()
        self.on_calls = 0
        self.on_I_samples = []

    def _online_add_power_sample(self, I_A):
        if I_A is None:
            return
        if len(self.on_I_samples) >= int(ONLINE_MAX_I_SAMPLES):
            return
        try:
            self.on_I_samples.append(float(I_A))
        except Exception:
            pass

    def _online_update_meter(self, frame_for_meter, results_list):
        if results_list is None:
            return
        try:
            r0 = results_list[0]
        except Exception:
            return
        try:
            self.on_meter.update(frame_for_meter, r0, time.time())
        except Exception:
            pass

    def _online_compute_score_rows(
        self,
        db,
        use_ctx,
        pin_bin,
        override_name=None,
        override_q=None,
        override_i=None,
        override_calls=None
    ):
        ctx = db.get("contexts", {}).get(use_ctx)
        if ctx is None:
            return []
        models = ctx.get("models", {})
        if not models:
            return []

        names = []
        q = []
        iA = []
        calls = []

        for name, rec in models.items():
            if rec is None:
                continue
            qq = rec.get("quality_score", None)
            cc = rec.get("infer_calls", None)
            if qq is None or cc is None:
                continue

            cur = rec.get("current_A", None)
            names.append(str(name))
            q.append(float(qq))
            calls.append(float(cc))
            iA.append(None if cur is None else float(cur))

        if not names:
            return []

        if override_name is not None and str(override_name) in names:
            idx = names.index(str(override_name))
            if override_q is not None:
                q[idx] = float(override_q)
            if override_calls is not None:
                calls[idx] = float(override_calls)
            if override_i is not None:
                iA[idx] = float(override_i)

        def minmax(vals, fill=None):
            arr = []
            for v in vals:
                arr.append(fill if v is None else v)
            mn = min(arr)
            mx = max(arr)
            if mx - mn < 1e-9:
                return [0.0 for _ in arr]
            return [(v - mn) / (mx - mn) for v in arr]

        valid_i = [v for v in iA if v is not None]
        if valid_i:
            worst = max(valid_i)
            i_norm = minmax(iA, fill=worst)
        else:
            i_norm = [0.0 for _ in iA]

        q_norm = minmax(q, fill=max(q))
        s_norm = minmax(calls, fill=min(calls))
        s_bad = [1.0 - v for v in s_norm]

        wi, wq, ws = weights_for_pin(pin_bin)

        rows = []
        for idx, name in enumerate(names):
            score = float(wi) * float(i_norm[idx]) + float(wq) * float(q_norm[idx]) + float(ws) * float(s_bad[idx])
            rows.append({"model": name, "score": float(score)})

        rows.sort(key=lambda r: r["score"])
        return rows

    def _online_maybe_commit_update(self, stable_ctx, cobj):
        if not ONLINE_ENABLE:
            return
        if not stable_ctx:
            return
        if self.in_crit:
            return
        if not self.shared.db_in_ram:
            return

        now = time.time()
        if self.on_win_t0 <= 0.0:
            return
        if (now - self.on_win_t0) < float(ONLINE_WIN_SEC):
            return
        if int(self.on_calls) < int(ONLINE_MIN_CALLS):
            return
        if (now - self.on_last_update_ts) < float(ONLINE_UPDATE_COOLDOWN_SEC):
            return

        mist_rate, conf_rate = self.on_meter.get_rates()
        q_online = 0.75 * float(mist_rate) + 0.25 * float(conf_rate)

        i_online = None
        if len(self.on_I_samples) >= 3:
            try:
                i_online = float(median(self.on_I_samples))
            except Exception:
                i_online = None
        elif len(self.on_I_samples) > 0:
            i_online = float(self.on_I_samples[len(self.on_I_samples) // 2])

        calls_online = int(self.on_calls)

        with self.shared.db_lock:
            use_cid = self.on_use_cid
            ctx = self.db.get("contexts", {}).get(use_cid)
            if ctx is None or not ctx.get("models", {}):
                return

            rows_db = self._online_compute_score_rows(self.db, use_cid, self.on_pin_bin)
            if len(rows_db) < int(ONLINE_RANK_CUTOFF):
                return

            rows_on = self._online_compute_score_rows(
                self.db,
                use_cid,
                self.on_pin_bin,
                override_name=self.on_model_name,
                override_q=q_online,
                override_i=i_online,
                override_calls=calls_online,
            )
            if not rows_on:
                return

            on_rank = None
            on_score = None
            for i, r in enumerate(rows_on):
                if r.get("model") == self.on_model_name:
                    on_rank = i + 1
                    on_score = float(r.get("score", 0.0))
                    break
            if on_rank is None:
                return

            cutoff_score = float(rows_db[int(ONLINE_RANK_CUTOFF) - 1].get("score", 0.0))
            if int(on_rank) <= int(ONLINE_RANK_CUTOFF):
                return
            if on_score < cutoff_score * (1.0 + float(ONLINE_HYST_REL)):
                return

            ctx_obj_for_write = cobj
            try:
                if "context" in ctx and isinstance(ctx.get("context"), dict):
                    ctx_obj_for_write = ctx.get("context")
            except Exception:
                pass

            # RAM-only update (no save)
            db_update_model(
                self.db,
                use_cid,
                ctx_obj_for_write,
                self.on_model_name,
                q_online,
                i_online,
                calls_online
            )

        with self.shared.status_lock:
            self.shared.db_in_ram = True
            self.shared.db_last_update_ts = time.time()
            self.shared.db_last_update_reason = "ONLINE"
            self.shared.db_dirty = True
            self.shared.db_state = "DIRTY(ONLINE_UPDATE)"

        self.on_last_update_ts = now

    # ---------- CRIT ----------
    def _enter_crit(self):
        # Disable benchmark immediately
        with self.shared.bench_ctrl_lock:
            self.shared.bench_enabled = False

        # Drain pending benchmark jobs so they cannot run/stamp later
        try:
            while True:
                job = self.shared.bench_queue.get_nowait()
                try:
                    ctx_id = job.get("context_id")
                except Exception:
                    ctx_id = None

                if ctx_id is not None:
                    try:
                        with self.shared.bench_pending_lock:
                            self.shared.bench_pending.discard(str(ctx_id))
                    except Exception:
                        pass

                try:
                    self.shared.bench_queue.task_done()
                except Exception:
                    pass
        except Exception:
            pass

        # Pick global best model BEFORE saving (only if DB still in RAM)
        crit_pick = None
        if self.shared.db_in_ram:
            with self.shared.db_lock:
                try:
                    stats = compute_global_model_stats(self.db)
                    crit_pick = pick_global_model_crit(stats)
                except Exception:
                    crit_pick = None

        crit_name = None
        if isinstance(crit_pick, dict):
            crit_name = crit_pick.get("model", None)

        if crit_name is not None:
            valid_names = [n for (n, _) in MODELS]
            if str(crit_name) in valid_names:
                try:
                    self.current_model = self._load_model(str(crit_name))
                    self.current_name = str(crit_name)
                    self.last_switch_ts = time.time()
                except Exception:
                    pass

        # Save DB to disk once on CRIT entry
        ok = True
        if self.shared.db_in_ram:
            with self.shared.db_lock:
                ok = db_save_atomic(self.db)

        # After saving, UNLOAD for real (clear contexts) if enabled
        did_unload = False
        if DB_UNLOAD_ON_CRIT:
            with self.shared.db_lock:
                try:
                    self.db.clear()
                    self.db.update(db_default())
                    did_unload = True
                except Exception:
                    did_unload = False

        with self.shared.status_lock:
            self.shared.sys_mode = "CRIT"
            self.shared.bench_state = "DISABLED"
            self.shared.db_last_save_ts = time.time()
            self.shared.db_dirty = False

            if did_unload:
                self.shared.db_in_ram = False
            else:
                self.shared.db_in_ram = True

            if ok:
                st = "SAVED(CRIT)"
            else:
                st = "SAVE_FAIL(CRIT)"
            if crit_name is not None:
                st = st + " pick=%s" % str(crit_name)
            if did_unload:
                st = st + " + UNLOADED(RAM)"
            self.shared.db_state = st

        self.in_crit = True

        # Reset current session lock (do not erase cache)
        self.lock_w = None
        self.calibrated = False
        self._res_stop_search()

    def _exit_crit(self):
        with self.shared.db_lock:
            newdb = db_load()
            _reload_db_inplace(self.db, newdb)

        with self.shared.bench_ctrl_lock:
            self.shared.bench_enabled = True

        with self.shared.status_lock:
            self.shared.sys_mode = "NORMAL"
            self.shared.bench_state = "ENABLED"
            self.shared.db_in_ram = True
            self.shared.db_last_update_ts = time.time()
            self.shared.db_last_update_reason = "RELOAD"
            self.shared.db_state = "RELOADED"
            self.shared.db_dirty = False

        self.in_crit = False
        self.lock_w = None
        self.calibrated = False
        self._res_stop_search()

    # ---------- RES (non-blocking) ----------
    def _res_cache_get(self, light_name):
        rec = self.res_cache.get(str(light_name))
        if rec is None:
            return None
        try:
            return int(rec.get("lock_w", None))
        except Exception:
            return None

    def _res_cache_put(self, light_name, lock_w):
        try:
            self.res_cache[str(light_name)] = {"lock_w": int(lock_w), "ts": time.time()}
        except Exception:
            pass

    def _res_stop_search(self):
        self.res_search_active = False
        self.res_search_light = None
        self.res_search_src = "LIVE"
        self.res_trial_w = None
        self.res_last_good_w = None
        self.res_good_streak = 0
        self.res_miss_streak = 0

    def _res_can_use_last(self):
        if self.last_obj_frame is None:
            return False
        if (time.time() - self.last_obj_ts) > LAST_OBJ_MAX_AGE_SEC:
            return False
        return True

    def _res_should_start_search(self, light_name, has_obj_live):
        if not self.in_crit or (not RES_ENABLE_ON_CRIT):
            return False
        if self.res_search_active:
            return False
        if self._res_cache_get(light_name) is not None:
            return False
        now = time.time()
        if (now - self.res_last_start_ts) < RES_MIN_RETEST_SEC:
            return False
        if has_obj_live:
            return True
        if self._res_can_use_last():
            return True
        return False

    def _res_start_search(self, light_name, has_obj_live):
        self.res_last_start_ts = time.time()
        self.res_search_active = True
        self.res_search_light = str(light_name)
        self.res_search_src = "LIVE" if has_obj_live else "LAST"
        self.res_trial_w = int(self.base_w)
        self.res_last_good_w = int(self.base_w)
        self.res_good_streak = 0
        self.res_miss_streak = 0

    def _res_finish_lock(self, light_name, lock_w, reason):
        self.lock_w = int(lock_w)
        self.calibrated = True
        self._res_cache_put(light_name, int(lock_w))
        self._res_stop_search()

    def _res_update_search_step(self, light_name, has_obj_at_trial):
        if not self.res_search_active:
            return
        if str(light_name) != str(self.res_search_light):
            self._res_stop_search()
            return

        if has_obj_at_trial:
            self.res_good_streak += 1
            self.res_miss_streak = 0
            if self.res_good_streak >= int(RES_GOOD_STREAK_N):
                self.res_last_good_w = int(self.res_trial_w)

                next_w = int(float(self.res_trial_w) * float(RES_SHRINK_RATIO))
                if next_w >= int(self.res_trial_w):
                    next_w = int(self.res_trial_w) - 1
                if next_w < int(RES_MIN_W):
                    return

                self.res_trial_w = int(next_w)
                self.res_good_streak = 0
                self.res_miss_streak = 0
        else:
            self.res_miss_streak += 1
            self.res_good_streak = 0
            if self.res_miss_streak >= int(RES_MISS_STREAK_N):
                lg = self.res_last_good_w
                if lg is None:
                    lg = int(self.base_w)
                self._res_finish_lock(light_name, int(lg), "OBJ_LOST")

    def run_forever(self):
        self._ensure_initial_model()

        while not self.shared.stop_flag:
            t0 = time.time()

            frame = self._get_frame_copy()
            if frame is None:
                time.sleep(0.01)
                continue

            if self.base_w is None:
                self.base_h, self.base_w = frame.shape[:2]

            pin_pct = None
            I_A_now = None
            if self.pwr is not None:
                I_A, pct, age = self.pwr.get_latest()
                I_A_now = I_A
                if pct is not None:
                    try:
                        pin_pct = float(pct)
                    except Exception:
                        pin_pct = None

            if pin_pct is not None:
                if (not self.in_crit) and (pin_pct <= CRIT_THRESHOLD):
                    self._enter_crit()
                elif self.in_crit and (pin_pct >= EXIT_CRIT_THRESHOLD):
                    self._exit_crit()

            now = time.time()
            cid, cobj, stable, light_name, dark_q, edge_q, pin_bin = self.ctx_tracker.update(frame, now, pin_pct)

            if cid != self.last_ctx_seen:
                self.last_ctx_seen = cid
                self.ctx_since = now
            stable_ctx = (self.ctx_since is not None) and ((now - self.ctx_since) >= CONTEXT_STABLE_SEC)

            with self.shared.bench_ctrl_lock:
                bench_allowed = bool(self.shared.bench_enabled)

            # only enqueue benchmark when DB is present in RAM
            if bench_allowed and stable_ctx and self.shared.db_in_ram:
                with self.shared.db_lock:
                    missing = (cid not in self.db.get("contexts", {}))

                if missing:
                    should_enqueue = False
                    try:
                        with self.shared.bench_pending_lock:
                            if str(cid) not in self.shared.bench_pending:
                                self.shared.bench_pending.add(str(cid))
                                should_enqueue = True
                    except Exception:
                        should_enqueue = True

                    if should_enqueue:
                        try:
                            self.shared.bench_queue.put_nowait({
                                "context_id": cid,
                                "context_obj": cobj
                            })
                        except Exception:
                            try:
                                with self.shared.bench_pending_lock:
                                    self.shared.bench_pending.discard(str(cid))
                            except Exception:
                                pass

            # choose model: if DB unloaded, fallback to first model
            use_cid = cid
            pick_name = None
            pick_score = None

            if self.shared.db_in_ram:
                with self.shared.db_lock:
                    use_cid = cid
                    if use_cid not in self.db.get("contexts", {}):
                        near = nearest_context_id(self.db, light_name, dark_q, edge_q, pin_bin)
                        if near is not None:
                            use_cid = near
                    pick_name, pick_score = choose_model_from_context(self.db, use_cid, pin_bin)

            if pick_name is None:
                pick_name = MODELS[0][0]
                pick_score = None

            if pick_name != self.current_name and (now - self.last_switch_ts) >= MIN_SWITCH_SEC:
                try:
                    self.current_model = self._load_model(pick_name)
                    self.current_name = pick_name
                    self.last_switch_ts = now
                except Exception:
                    pass

            if ONLINE_ENABLE and self.shared.db_in_ram and (not self.in_crit):
                if (
                    (self.on_model_name != str(self.current_name)) or
                    (self.on_use_cid != str(use_cid)) or
                    (self.on_pin_bin != str(pin_bin))
                ):
                    self._online_reset(use_cid, pin_bin, self.current_name)

            cur_w = int(self.base_w)

            cached_w = None
            if self.in_crit and RES_ENABLE_ON_CRIT:
                cached_w = self._res_cache_get(light_name)

                if (cached_w is not None) and (not self.res_search_active):
                    self.lock_w = int(cached_w)
                    self.calibrated = True

                if self.res_search_active and (self.res_trial_w is not None):
                    cur_w = int(self.res_trial_w)
                elif self.calibrated and (self.lock_w is not None):
                    cur_w = int(self.lock_w)

            infer_src = "LIVE"
            fr_src = frame
            if self.in_crit and RES_ENABLE_ON_CRIT and self.res_search_active:
                if self.res_search_src == "LAST" and self._res_can_use_last():
                    infer_src = "LAST"
                    fr_src = self.last_obj_frame.copy()

            fr_in = fr_src
            if cur_w < int(self.base_w):
                fr_in = resize_keep_aspect(fr_src, cur_w)

            annotated_rs, r = self._infer(fr_in)

            if ONLINE_ENABLE and self.shared.db_in_ram and (not self.in_crit):
                self.on_calls += 1
                self._online_add_power_sample(I_A_now)
                self._online_update_meter(fr_in, r)

            has_obj, conf_max, n_boxes = self._has_obj(r, CONF_TH)

            if has_obj and infer_src == "LIVE":
                self.last_obj_frame = frame.copy()
                self.last_obj_ts = time.time()

            if self._res_should_start_search(light_name, has_obj_live=(has_obj and infer_src == "LIVE")):
                self._res_start_search(light_name, has_obj_live=(has_obj and infer_src == "LIVE"))

            if self.res_search_active:
                self._res_update_search_step(light_name, has_obj_at_trial=bool(has_obj))

            if ONLINE_ENABLE and self.shared.db_in_ram:
                self._online_maybe_commit_update(stable_ctx=stable_ctx, cobj=cobj)
                if (time.time() - self.on_win_t0) >= float(ONLINE_WIN_SEC):
                    self._online_reset(use_cid, pin_bin, self.current_name)

            if cur_w < int(self.base_w):
                out = make_canvas(self.base_h, self.base_w, annotated_rs)
            else:
                out = annotated_rs

            # ===== overlay: keep old form =====
            try:
                cv2.putText(out, "CTX: %s" % str(cid), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
                cv2.putText(out, "USE_CTX: %s" % str(use_cid), (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
                cv2.putText(out, "MODEL: %s" % str(self.current_name), (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

                mode_s = "CRIT" if self.in_crit else "NORMAL"
                cv2.putText(out, "MODE: %s" % mode_s, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

                if pin_pct is not None:
                    cv2.putText(
                        out,
                        "PIN: %.1f%% (%s)" % (float(pin_pct), str(pin_bin)),
                        (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2
                    )
                else:
                    cv2.putText(
                        out,
                        "PIN: NA (%s)" % str(pin_bin),
                        (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2
                    )

                if self.in_crit and RES_ENABLE_ON_CRIT:
                    hit = "HIT" if (cached_w is not None) else "MISS"
                    rs = "SEARCH" if self.res_search_active else "LOCK" if (self.calibrated and self.lock_w is not None) else "NONE"
                    src = infer_src
                    lw = self.lock_w if self.lock_w is not None else -1
                    tw = self.res_trial_w if self.res_trial_w is not None else -1

                    cv2.putText(
                        out,
                        "RES light=%s cache=%s state=%s src=%s" % (str(light_name), hit, rs, str(src)),
                        (10, 152),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2
                    )
                    cv2.putText(
                        out,
                        "RES cur_w=%d lock_w=%d trial_w=%d" % (int(cur_w), int(lw), int(tw)),
                        (10, 174),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2
                    )

                with self.shared.status_lock:
                    in_ram = bool(getattr(self.shared, "db_in_ram", True))
                    dirty = bool(getattr(self.shared, "db_dirty", False))
                    last_upd_ts = float(getattr(self.shared, "db_last_update_ts", 0.0))
                    last_upd_reason = str(getattr(self.shared, "db_last_update_reason", "NA"))
                    last_save_ts = float(getattr(self.shared, "db_last_save_ts", 0.0))
                    loaded_ts = float(getattr(self.shared, "db_loaded_ts", 0.0))

                try:
                    ctx_count = int(len(self.db.get("contexts", {})))
                except Exception:
                    ctx_count = -1

                cv2.putText(
                    out,
                    "DB in_ram=%d dirty=%d ctxs=%d" % (1 if in_ram else 0, 1 if dirty else 0, int(ctx_count)),
                    (10, 196),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    out,
                    "DB last_upd=%s (%s)" % (_fmt_hms(last_upd_ts), str(last_upd_reason)),
                    (10, 218),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    out,
                    "DB last_save=%s loaded=%s" % (_fmt_hms(last_save_ts), _fmt_hms(loaded_ts)),
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 255),
                    2
                )

            except Exception:
                pass

            try:
                cv2.imshow("Realtime", out)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord("q"):
                    self.shared.stop_flag = True
            except Exception:
                pass

            dt_ms = (time.time() - t0) * 1000.0
            with self.shared.rt_lock:
                self.shared.rt_last_ms = float(dt_ms)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass