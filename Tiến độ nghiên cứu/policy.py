# policy.py
# ASCII-only.

import numpy as np


def weights_for_pin(pin_bin):
    # (wi, wq, ws) -> power, quality, speed
    if pin_bin in ["CRIT", "LOW"]:
        return (0.55, 0.30, 0.15)  # power > quality > speed
    return (0.20, 0.55, 0.25)      # quality > speed > power


def parse_context_id(cid):
    try:
        parts = cid.split("__")
        light = parts[0]
        dark_q = int(parts[1].replace("dark", ""))
        edge_q = int(parts[2].replace("edge", ""))
        pin = parts[3]
        return light, dark_q, edge_q, pin
    except Exception:
        return None


def nearest_context_id(db, light_name, dark_q, edge_q, pin_bin):
    ctxs = db.get("contexts", {})
    if not ctxs:
        return None

    cand = []
    for cid in ctxs.keys():
        p = parse_context_id(cid)
        if p is None:
            continue
        l, dq, eq, pb = p
        cand.append((cid, l, dq, eq, pb))

    stages = [
        lambda x: (x[1] == light_name and x[4] == pin_bin),
        lambda x: (x[1] == light_name),
        lambda x: (x[4] == pin_bin),
        lambda x: True
    ]

    for stage in stages:
        best = None
        best_dist = 1e18

        for (cid, l, dq, eq, pb) in cand:
            if not stage((cid, l, dq, eq, pb)):
                continue

            dist = abs(int(dark_q) - int(dq)) + abs(int(edge_q) - int(eq))

            if l != light_name:
                dist += 30

            if pb != pin_bin:
                dist += 10

            if dist < best_dist:
                best_dist = dist
                best = cid

        if best is not None:
            return best

    return None


def _choose_model_from_cached_score(models, pin_bin):
    best_name = None
    best_score = None

    for name, rec in models.items():
        if rec is None:
            continue

        cache = rec.get("score_cache", None)
        if not isinstance(cache, dict):
            return None, None

        score = cache.get(str(pin_bin), None)

        # If exact pin_bin is missing, use HIGH as normal fallback.
        if score is None:
            score = cache.get("HIGH", None)

        if score is None:
            return None, None

        try:
            score = float(score)
        except Exception:
            return None, None

        if best_score is None or score < best_score:
            best_score = score
            best_name = name

    return best_name, best_score


def _choose_model_slow_fallback(models, pin_bin):
    """
    Old behavior fallback.
    Used only when DB has no score_cache yet.
    Lower score wins.
    """
    names = []
    q_list = []
    i_list = []
    s_list = []

    for name, rec in models.items():
        q = rec.get("quality_score", None)
        iA = rec.get("current_A", None)
        calls = rec.get("infer_calls", None)

        if q is None or calls is None:
            continue

        names.append(name)
        q_list.append(float(q))
        s_list.append(float(calls))
        i_list.append(float(iA) if iA is not None else float("nan"))

    if not names:
        return None, None

    q = np.array(q_list, dtype=np.float32)
    s = np.array(s_list, dtype=np.float32)
    iA = np.array(i_list, dtype=np.float32)

    def norm_minmax(a, nan_fill=None):
        a2 = a.copy()

        if nan_fill is not None:
            a2[np.isnan(a2)] = nan_fill

        mn = float(np.min(a2))
        mx = float(np.max(a2))

        if mx - mn < 1e-9:
            return np.zeros_like(a2, dtype=np.float32)

        return (a2 - mn) / (mx - mn)

    if np.all(np.isnan(iA)):
        i_norm = np.zeros_like(iA, dtype=np.float32)
    else:
        valid = iA[~np.isnan(iA)]
        worst = float(np.max(valid))
        i_norm = norm_minmax(iA, nan_fill=worst)

    q_norm = norm_minmax(q, nan_fill=float(np.max(q)))
    s_norm = norm_minmax(s, nan_fill=float(np.min(s)))
    s_bad = 1.0 - s_norm

    wi, wq, ws = weights_for_pin(pin_bin)
    score = wi * i_norm + wq * q_norm + ws * s_bad

    bi = int(np.argmin(score))
    return names[bi], float(score[bi])


def choose_model_from_context(db, context_id, pin_bin):
    ctx = db.get("contexts", {}).get(context_id)
    if ctx is None:
        return None, None

    models = ctx.get("models", {})
    if not models:
        return None, None

    # Fast path:
    # Use precomputed score_cache from DB.
    pick_name, pick_score = _choose_model_from_cached_score(models, pin_bin)
    if pick_name is not None:
        return pick_name, float(pick_score)

    # Safe fallback:
    # If old DB has no score_cache, still works like before.
    return _choose_model_slow_fallback(models, pin_bin)