# debug_window.py
# ASCII-only.

import time
import cv2
import numpy as np


def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return "NA"


def _fmt_float(x, nd=3, na="NA"):
    if x is None:
        return na
    try:
        return ("%." + str(nd) + "f") % float(x)
    except Exception:
        return na


def _draw_text(img, x, y, s, scale=0.52, thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)


def _collect_ranking(db, use_ctx, pin_bin):
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
        if rec.get("quality_score") is None or rec.get("infer_calls") is None:
            continue

        names.append(name)
        q.append(float(rec.get("quality_score", 0.0)))
        calls.append(float(rec.get("infer_calls", 0.0)))

        cur = rec.get("current_A", None)
        iA.append(None if cur is None else float(cur))

    if not names:
        return []

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

    if pin_bin in ["CRIT", "LOW"]:
        wi, wq, ws = (0.55, 0.30, 0.15)
    else:
        wi, wq, ws = (0.20, 0.55, 0.25)

    rows = []
    for idx, name in enumerate(names):
        score = wi * i_norm[idx] + wq * q_norm[idx] + ws * s_bad[idx]
        rows.append({
            "model": name,
            "score": float(score),
            "quality": float(q[idx]),
            "current": iA[idx],
            "calls": int(calls[idx]),
        })

    rows.sort(key=lambda r: r["score"])
    return rows


class DebugWindow:
    def __init__(self, shared, db_ref):
        self.shared = shared
        self.db = db_ref
        self.win = "Debug"

        self.w = 1080
        self.h = 700

    def run_forever(self):
        last_ctx = ""
        last_pin_bin = "NA"

        while not self.shared.stop_flag:
            img = np.full((self.h, self.w, 3), 255, dtype=np.uint8)

            with self.shared.status_lock:
                sys_mode = self.shared.sys_mode
                bench_state = self.shared.bench_state
                db_state = self.shared.db_state

            with self.shared.rt_lock:
                rt_ms = float(self.shared.rt_last_ms)

            with self.shared.res_lock:
                res_status = self.shared.res_status
                res_light = self.shared.res_calib_light
                res_running = bool(self.shared.res_calib_running)
                res_active = self.shared.res_active_lock_w
                cache = dict(self.shared.res_lockw_by_light)

            # Try to get a "current context" from cache in a cheap way:
            # We don't have ctx_id in shared, so we display only system status + RES.
            # (Realtime overlay already shows ctx/use/model.)

            _draw_text(img, 20, 30, "SYSTEM")
            _draw_text(img, 20, 55, "sys_mode : %s" % _safe_str(sys_mode))
            _draw_text(img, 20, 80, "bench    : %s" % _safe_str(bench_state))
            _draw_text(img, 20, 105, "db_state : %s" % _safe_str(db_state))
            _draw_text(img, 20, 130, "rt_loop  : %s ms" % _fmt_float(rt_ms, 1))

            _draw_text(img, 20, 170, "RES")
            _draw_text(img, 20, 195, "status   : %s" % _safe_str(res_status))
            _draw_text(img, 20, 220, "running  : %s light=%s" % (_safe_str(res_running), _safe_str(res_light)))
            _draw_text(img, 20, 245, "active_w : %s" % (_safe_str(res_active) if res_active is not None else "NA"))

            y = 280
            _draw_text(img, 20, y, "cache_by_light:")
            y += 25
            for k in ["BRIGHT", "NORMAL", "DIM", "DARK"]:
                v = cache.get(k)
                _draw_text(img, 40, y, "%-6s : %s" % (k, ("NA" if v is None else str(int(v)))))
                y += 22

            # queue size
            try:
                qsz = self.shared.res_queue.qsize()
            except Exception:
                qsz = -1
            _draw_text(img, 20, y + 10, "res_queue_size: %d" % int(qsz))

            cv2.imshow(self.win, img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):
                self.shared.stop_flag = True

            time.sleep(0.03)
