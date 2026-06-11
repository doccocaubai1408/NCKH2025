# db_benchmark.py
# ASCII-only.

import os
import json
import time

from config import DB_PATH


PIN_SCORE_BINS = ["CRIT", "LOW", "MID", "HIGH", "NA"]


def db_default():
    return {
        "meta": {
            "schema_version": 3,
            "updated_at": ""
        },
        "contexts": {}
    }


def _weights_for_pin_cache(pin_bin):
    # Keep same meaning as policy.weights_for_pin()
    # (wi, wq, ws) -> power, quality, speed
    if pin_bin in ["CRIT", "LOW"]:
        return (0.55, 0.30, 0.15)
    return (0.20, 0.55, 0.25)


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


def _minmax(vals, fill=None):
    arr = []
    for v in vals:
        arr.append(fill if v is None else v)

    if not arr:
        return []

    mn = min(arr)
    mx = max(arr)

    if mx - mn < 1e-9:
        return [0.0 for _ in arr]

    return [(v - mn) / (mx - mn) for v in arr]


def recompute_context_score_cache(db, context_id):
    """
    Precompute score_cache for one context.

    Writes:
      rec["score_cache"][pin_bin] = score

    Lower score is better.

    This does NOT change quality/current/calls.
    It only adds cached decision scores.
    """
    try:
        ctx = db.get("contexts", {}).get(context_id)
        if ctx is None:
            return False

        models = ctx.get("models", {})
        if not models:
            return False

        names = []
        q_list = []
        i_list = []
        calls_list = []

        for name, rec in models.items():
            if rec is None:
                continue

            q = _safe_float(rec.get("quality_score", None), None)
            calls = _safe_float(rec.get("infer_calls", None), None)
            cur = _safe_float(rec.get("current_A", None), None)

            # Need quality and calls to make a meaningful score.
            if q is None or calls is None:
                continue

            names.append(str(name))
            q_list.append(float(q))
            calls_list.append(float(calls))
            i_list.append(cur)

        if not names:
            return False

        valid_i = [v for v in i_list if v is not None]

        if valid_i:
            worst_i = max(valid_i)
            i_norm = _minmax(i_list, fill=worst_i)
        else:
            i_norm = [0.0 for _ in i_list]

        q_norm = _minmax(q_list, fill=max(q_list))
        s_norm = _minmax(calls_list, fill=min(calls_list))

        # More infer_calls = faster/better, so convert to badness.
        s_bad = [1.0 - v for v in s_norm]

        score_by_model = {}
        for pin_bin in PIN_SCORE_BINS:
            wi, wq, ws = _weights_for_pin_cache(pin_bin)

            for idx, name in enumerate(names):
                score = (
                    float(wi) * float(i_norm[idx])
                    + float(wq) * float(q_norm[idx])
                    + float(ws) * float(s_bad[idx])
                )

                if name not in score_by_model:
                    score_by_model[name] = {}

                score_by_model[name][pin_bin] = float(score)

        for name in names:
            rec = models.get(name)
            if rec is None:
                continue
            rec["score_cache"] = dict(score_by_model.get(name, {}))

        return True

    except Exception:
        return False


def recompute_all_score_cache(db):
    try:
        ctxs = db.get("contexts", {})
        for cid in list(ctxs.keys()):
            recompute_context_score_cache(db, cid)
        return True
    except Exception:
        return False


def db_load(path=DB_PATH):
    if not os.path.exists(path):
        return db_default()

    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        if "contexts" not in j:
            return db_default()

        if "meta" not in j:
            j["meta"] = {}

        # Upgrade old DB in RAM.
        j["meta"]["schema_version"] = 3
        recompute_all_score_cache(j)

        return j

    except Exception:
        return db_default()


def db_save_atomic(db, path=DB_PATH):
    try:
        if "meta" not in db:
            db["meta"] = {}

        db["meta"]["schema_version"] = 3
        db["meta"]["updated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.localtime()
        )

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)

        os.replace(tmp, path)
        return True

    except Exception as e:
        print("DB save failed:", str(e))
        return False


def db_update_model(
    db,
    context_id,
    context_obj,
    model_name,
    quality_score,
    current_A,
    infer_calls
):
    ctxs = db["contexts"]

    if context_id not in ctxs:
        ctxs[context_id] = {
            "context": dict(context_obj),
            "models": {}
        }

    m = ctxs[context_id]["models"].get(model_name)

    if m is None:
        ctxs[context_id]["models"][model_name] = {
            "quality_score": float(quality_score),
            "current_A": None if current_A is None else float(current_A),
            "infer_calls": int(infer_calls),
            "samples": 1,
            "score_cache": {}
        }

        recompute_context_score_cache(db, context_id)
        return

    samples = int(m.get("samples", 1))
    alpha = 1.0 / float(min(10, samples + 1))

    m["quality_score"] = (
        (1.0 - alpha) * float(m.get("quality_score", 0.0))
        + alpha * float(quality_score)
    )

    if current_A is not None:
        if m.get("current_A") is None:
            m["current_A"] = float(current_A)
        else:
            m["current_A"] = (
                (1.0 - alpha) * float(m.get("current_A", 0.0))
                + alpha * float(current_A)
            )

    m["infer_calls"] = int(round(
        (1.0 - alpha) * float(m.get("infer_calls", 0))
        + alpha * float(infer_calls)
    ))

    m["samples"] = samples + 1

    # Important:
    # After any learning/online update, refresh cached scores for this context.
    recompute_context_score_cache(db, context_id)