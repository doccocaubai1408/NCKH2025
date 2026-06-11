# power_mode.py
# ASCII-only.

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

def compute_global_model_stats(db):
    """
    Aggregate across ALL contexts to get per-model averages:
    - avg current_A (lower better)
    - avg quality_score (lower better)
    - avg infer_calls (higher better)
    Returns dict: {model_name: {"I":avg, "Q":avg, "C":avg, "n":count}}
    """
    out = {}
    ctxs = db.get("contexts", {})
    for _, ctx in ctxs.items():
        models = ctx.get("models", {})
        for name, rec in models.items():
            if rec is None:
                continue
            I = _safe_float(rec.get("current_A", None), None)
            Q = _safe_float(rec.get("quality_score", None), None)
            C = _safe_float(rec.get("infer_calls", None), None)

            # Require at least I and Q to be meaningful for CRIT pick
            if I is None or Q is None:
                continue

            if name not in out:
                out[name] = {"sumI": 0.0, "sumQ": 0.0, "sumC": 0.0, "n": 0}
            out[name]["sumI"] += float(I)
            out[name]["sumQ"] += float(Q)
            out[name]["sumC"] += float(C) if C is not None else 0.0
            out[name]["n"] += 1

    stats = {}
    for name, a in out.items():
        n = max(1, int(a["n"]))
        stats[name] = {
            "I": a["sumI"] / n,
            "Q": a["sumQ"] / n,
            "C": a["sumC"] / n,
            "n": n,
        }
    return stats

def _minmax_norm(values, invert=False):
    # values: list of floats
    mn = min(values)
    mx = max(values)
    if mx - mn < 1e-9:
        norm = [0.0 for _ in values]
    else:
        norm = [(v - mn) / (mx - mn) for v in values]
    if invert:
        # higher better -> invert to penalty (lower better)
        norm = [1.0 - v for v in norm]
    return norm

def pick_global_model_crit(stats):
    """
    pin priority > accuracy > speed.
    Score uses normalized penalties:
      score = 0.65*I_pen + 0.25*Q_pen + 0.10*Speed_pen
    Lower score wins.
    """
    if not stats:
        return None

    names = list(stats.keys())
    Is = [float(stats[n]["I"]) for n in names]
    Qs = [float(stats[n]["Q"]) for n in names]
    Cs = [float(stats[n]["C"]) for n in names]

    I_pen = _minmax_norm(Is, invert=False)   # low better
    Q_pen = _minmax_norm(Qs, invert=False)   # low better
    S_pen = _minmax_norm(Cs, invert=True)    # high calls better -> penalty invert

    wi, wq, ws = (0.65, 0.25, 0.10)

    best = None
    best_score = None
    for i, name in enumerate(names):
        score = wi * I_pen[i] + wq * Q_pen[i] + ws * S_pen[i]
        if best_score is None or score < best_score:
            best_score = score
            best = name

    return {"model": best, "score": float(best_score), "stats": stats.get(best, {})}

def drop_db_contexts_keep_header(db):
    """
    Optional RAM free: keep db structure but drop contexts.
    """
    if "contexts" in db:
        db["contexts"] = {}
