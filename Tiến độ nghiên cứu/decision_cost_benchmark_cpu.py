# decision_cost_benchmark_cpu.py
# ASCII-only.
#
# Standalone benchmark for decision cost only, matched to NEW system-1 logic:
#
#   context + pin -> exact/nearest DB lookup -> choose_model_from_context()
#
# NEW decision logic:
#   - Fast path: choose_model_from_context() reads precomputed rec["score_cache"][pin_bin]
#   - Slow fallback: if score_cache is missing/invalid, it recomputes old score
#
# This benchmark measures BOTH:
#   - wall time (ms/decision)
#   - CPU percent like learning/BenchmarkWorker style
#
# It does NOT run:
#   camera, serial, YOLO inference, benchmark learning,
#   online update, adaptive resolution, or DB saving.

import csv
import random
import statistics
import time

import psutil

from config import MODELS
from db_benchmark import db_load
from policy import nearest_context_id, choose_model_from_context, parse_context_id


RUNS = 1000
BATCH_SIZE = 100

OUT_CSV = "decision_cost_benchmark_cpu.csv"
OUT_TXT = "decision_cost_summary_cpu.txt"

RANDOM_SEED = 123
INCLUDE_NEAR_CASES = True

PIN_BINS = ["CRIT", "LOW", "MID", "HIGH", "NA"]
LIGHTS = ["BRIGHT", "NORMAL", "DIM", "DARK"]

process = psutil.Process()
NUM_CORES = psutil.cpu_count(logical=True) or 1


def make_context_id(light_name, dark_q, edge_q, pin_bin):
    return "%s__dark%03d__edge%03d__%s" % (
        str(light_name),
        int(dark_q),
        int(edge_q),
        str(pin_bin),
    )


def clamp_q(x):
    x = int(x)
    if x < 0:
        x = 0
    if x > 100:
        x = 100
    return int((x // 5) * 5)


def has_score_cache_for_pin(db, context_id, pin_bin):
    """
    Detect whether NEW fast path should be usable.
    This only labels the result; it does not change decision behavior.
    """
    try:
        ctx = db.get("contexts", {}).get(context_id)
        if ctx is None:
            return False

        models = ctx.get("models", {})
        if not models:
            return False

        checked = 0

        for _name, rec in models.items():
            if rec is None:
                continue

            if rec.get("quality_score", None) is None:
                continue

            if rec.get("infer_calls", None) is None:
                continue

            checked += 1

            cache = rec.get("score_cache", None)
            if not isinstance(cache, dict):
                return False

            score = cache.get(str(pin_bin), None)
            if score is None:
                score = cache.get("HIGH", None)

            if score is None:
                return False

            try:
                float(score)
            except Exception:
                return False

        return checked > 0

    except Exception:
        return False


def build_test_cases(db, runs):
    ctxs = db.get("contexts", {})
    ids = list(ctxs.keys())

    if not ids:
        raise RuntimeError("DB has no contexts. Run learning/benchmark first.")

    cases = []

    for i in range(int(runs)):
        base_cid = random.choice(ids)
        parsed = parse_context_id(base_cid)

        if parsed is None:
            light_name = random.choice(LIGHTS)
            dark_q = random.randrange(0, 101, 5)
            edge_q = random.randrange(0, 101, 5)
            pin_bin = random.choice(PIN_BINS)
        else:
            light_name, dark_q, edge_q, pin_bin = parsed

        if INCLUDE_NEAR_CASES and (i % 2 == 1):
            dark_q = clamp_q(int(dark_q) + random.choice([-10, -5, 5, 10]))
            edge_q = clamp_q(int(edge_q) + random.choice([-10, -5, 5, 10]))

            if random.random() < 0.35:
                pin_bin = random.choice(PIN_BINS)

            cid = make_context_id(light_name, dark_q, edge_q, pin_bin)
            case_kind = "NEAR_OR_MISSING_INPUT"
        else:
            cid = base_cid
            case_kind = "EXACT_INPUT"

        cases.append({
            "case_index": i + 1,
            "case_kind": case_kind,
            "context_id": cid,
            "light_name": light_name,
            "dark_q": int(dark_q),
            "edge_q": int(edge_q),
            "pin_bin": str(pin_bin),
        })

    return cases


def decide_once(db, cid, light_name, dark_q, edge_q, pin_bin):
    """
    Mirrors realtime_worker.py decision block.
    choose_model_from_context() now uses score_cache fast path when available.
    """
    use_cid = cid
    pick_name = None
    pick_score = None
    decision_type = "FALLBACK"

    if use_cid in db.get("contexts", {}):
        decision_type = "EXACT"
    else:
        near = nearest_context_id(db, light_name, dark_q, edge_q, pin_bin)
        if near is not None:
            use_cid = near
            decision_type = "NEAREST"

    fast_cache_available = has_score_cache_for_pin(db, use_cid, pin_bin)

    pick_name, pick_score = choose_model_from_context(db, use_cid, pin_bin)

    if pick_name is None:
        pick_name = MODELS[0][0]
        pick_score = None
        decision_type = "FALLBACK"
        decision_path = "MODEL_FALLBACK"
    else:
        decision_path = "FAST_SCORE_CACHE" if fast_cache_available else "SLOW_SCORE_FALLBACK"

    return use_cid, decision_type, decision_path, pick_name, pick_score


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0

    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    k = (len(sorted_vals) - 1) * (float(p) / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)

    if f == c:
        return float(sorted_vals[f])

    return float(sorted_vals[f]) * (c - k) + float(sorted_vals[c]) * (k - f)


def cpu_percent_from_times(cpu_before, cpu_after, wall_time_sec):
    cpu_time_sec = (
        cpu_after.user - cpu_before.user
        + cpu_after.system - cpu_before.system
    )

    cpu_total_percent = (
        (cpu_time_sec / wall_time_sec) * 100.0
        if wall_time_sec > 0 else 0.0
    )

    cpu_machine_percent = cpu_total_percent / float(max(1, NUM_CORES))

    return cpu_time_sec, cpu_total_percent, cpu_machine_percent


def main():
    random.seed(RANDOM_SEED)

    db = db_load()
    cases = build_test_cases(db, RUNS)

    rows = []
    batch_rows = []
    durations_ms = []

    exact_count = 0
    nearest_count = 0
    fallback_count = 0

    fast_cache_count = 0
    slow_score_fallback_count = 0
    model_fallback_count = 0

    warm_n = min(50, len(cases))
    for c in cases[:warm_n]:
        decide_once(
            db,
            c["context_id"],
            c["light_name"],
            c["dark_q"],
            c["edge_q"],
            c["pin_bin"],
        )

    all_cpu_before = process.cpu_times()
    all_t0 = time.perf_counter()

    batch_index = 0
    i = 0

    while i < len(cases):
        batch_index += 1
        batch_cases = cases[i:i + int(BATCH_SIZE)]
        batch_n = len(batch_cases)

        batch_cpu_before = process.cpu_times()
        batch_t0 = time.perf_counter()

        for c in batch_cases:
            one_t0 = time.perf_counter()

            use_cid, decision_type, decision_path, pick_name, pick_score = decide_once(
                db,
                c["context_id"],
                c["light_name"],
                c["dark_q"],
                c["edge_q"],
                c["pin_bin"],
            )

            one_t1 = time.perf_counter()

            dms = (one_t1 - one_t0) * 1000.0
            durations_ms.append(dms)

            if decision_type == "EXACT":
                exact_count += 1
            elif decision_type == "NEAREST":
                nearest_count += 1
            else:
                fallback_count += 1

            if decision_path == "FAST_SCORE_CACHE":
                fast_cache_count += 1
            elif decision_path == "SLOW_SCORE_FALLBACK":
                slow_score_fallback_count += 1
            else:
                model_fallback_count += 1

            rows.append([
                c["case_index"],
                c["case_kind"],
                c["context_id"],
                use_cid,
                c["pin_bin"],
                "%.9f" % float(dms),
                decision_type,
                decision_path,
                pick_name,
                "NA" if pick_score is None else "%.9f" % float(pick_score),
            ])

        batch_t1 = time.perf_counter()
        batch_cpu_after = process.cpu_times()

        batch_wall_sec = batch_t1 - batch_t0
        batch_cpu_sec, batch_cpu_total, batch_cpu_machine = cpu_percent_from_times(
            batch_cpu_before,
            batch_cpu_after,
            batch_wall_sec,
        )

        batch_rows.append({
            "batch_index": batch_index,
            "batch_runs": batch_n,
            "batch_wall_sec": batch_wall_sec,
            "batch_avg_time_ms": (batch_wall_sec / float(max(1, batch_n))) * 1000.0,
            "batch_cpu_time_sec": batch_cpu_sec,
            "batch_cpu_total_percent": batch_cpu_total,
            "batch_cpu_machine_percent": batch_cpu_machine,
        })

        i += int(BATCH_SIZE)

    all_t1 = time.perf_counter()
    all_cpu_after = process.cpu_times()

    total_wall_sec = all_t1 - all_t0
    total_cpu_sec, cpu_total_percent, cpu_machine_percent = cpu_percent_from_times(
        all_cpu_before,
        all_cpu_after,
        total_wall_sec,
    )

    durations_sorted = sorted(durations_ms)
    avg_ms = statistics.mean(durations_ms) if durations_ms else 0.0
    med_ms = statistics.median(durations_ms) if durations_ms else 0.0
    min_ms = min(durations_ms) if durations_ms else 0.0
    max_ms = max(durations_ms) if durations_ms else 0.0
    p95_ms = percentile(durations_sorted, 95)
    p99_ms = percentile(durations_sorted, 99)

    batch_cpu_total_vals = [r["batch_cpu_total_percent"] for r in batch_rows]
    batch_cpu_machine_vals = [r["batch_cpu_machine_percent"] for r in batch_rows]

    avg_batch_cpu_total = statistics.mean(batch_cpu_total_vals) if batch_cpu_total_vals else 0.0
    avg_batch_cpu_machine = statistics.mean(batch_cpu_machine_vals) if batch_cpu_machine_vals else 0.0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case_index",
            "case_kind",
            "input_context_id",
            "use_context_id",
            "pin_bin",
            "decision_duration_ms",
            "decision_type",
            "decision_path",
            "picked_model",
            "picked_score",
        ])
        for r in rows:
            w.writerow(r)

    batch_csv = OUT_CSV.replace(".csv", "_batches.csv")
    with open(batch_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "batch_index",
            "batch_runs",
            "batch_wall_sec",
            "batch_avg_time_ms",
            "batch_cpu_time_sec",
            "batch_cpu_total_percent",
            "batch_cpu_machine_percent",
        ])
        for r in batch_rows:
            w.writerow([
                int(r["batch_index"]),
                int(r["batch_runs"]),
                "%.9f" % float(r["batch_wall_sec"]),
                "%.9f" % float(r["batch_avg_time_ms"]),
                "%.9f" % float(r["batch_cpu_time_sec"]),
                "%.6f" % float(r["batch_cpu_total_percent"]),
                "%.6f" % float(r["batch_cpu_machine_percent"]),
            ])

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("decision cost benchmark with CPU\n")
        f.write("================================\n")
        f.write("logic: score_cache_fast_path\n")
        f.write("runs: %d\n" % int(RUNS))
        f.write("batch_size: %d\n" % int(BATCH_SIZE))
        f.write("cpu_cores: %d\n" % int(NUM_CORES))
        f.write("total_wall_sec: %.9f\n" % float(total_wall_sec))
        f.write("total_cpu_time_sec: %.9f\n" % float(total_cpu_sec))
        f.write("avg_decision_duration_ms: %.9f\n" % float(avg_ms))
        f.write("median_decision_duration_ms: %.9f\n" % float(med_ms))
        f.write("p95_decision_duration_ms: %.9f\n" % float(p95_ms))
        f.write("p99_decision_duration_ms: %.9f\n" % float(p99_ms))
        f.write("min_decision_duration_ms: %.9f\n" % float(min_ms))
        f.write("max_decision_duration_ms: %.9f\n" % float(max_ms))
        f.write("cpu_total_percent: %.6f\n" % float(cpu_total_percent))
        f.write("cpu_machine_percent: %.6f\n" % float(cpu_machine_percent))
        f.write("avg_batch_cpu_total_percent: %.6f\n" % float(avg_batch_cpu_total))
        f.write("avg_batch_cpu_machine_percent: %.6f\n" % float(avg_batch_cpu_machine))
        f.write("exact_count: %d\n" % int(exact_count))
        f.write("nearest_count: %d\n" % int(nearest_count))
        f.write("fallback_count: %d\n" % int(fallback_count))
        f.write("fast_score_cache_count: %d\n" % int(fast_cache_count))
        f.write("slow_score_fallback_count: %d\n" % int(slow_score_fallback_count))
        f.write("model_fallback_count: %d\n" % int(model_fallback_count))
        f.write("csv_file: %s\n" % str(OUT_CSV))
        f.write("batch_csv_file: %s\n" % str(batch_csv))

    print("Done.")
    print("Wrote:", OUT_CSV)
    print("Wrote:", batch_csv)
    print("Wrote:", OUT_TXT)
    print("avg_ms=%.9f median_ms=%.9f p95_ms=%.9f p99_ms=%.9f" % (
        float(avg_ms), float(med_ms), float(p95_ms), float(p99_ms)
    ))
    print("CPU total=%.6f%% CPU machine=%.6f%% cores~=%.6f" % (
        float(cpu_total_percent),
        float(cpu_machine_percent),
        float(cpu_total_percent) / 100.0,
    ))
    print("fast_score_cache_count=%d slow_score_fallback_count=%d model_fallback_count=%d" % (
        int(fast_cache_count),
        int(slow_score_fallback_count),
        int(model_fallback_count),
    ))


if __name__ == "__main__":
    main()
