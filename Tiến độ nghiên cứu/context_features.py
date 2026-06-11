# context_features.py
# ASCII-only.

import cv2
import numpy as np

from config import (
    ANALYZE_W, ANALYZE_H, DARK_PIXEL_Y, EMA_ALPHA,
    MEAN_BINS, DARK_BINS, STABLE_HOLD_SEC,
    EDGE_W, EDGE_H, CANNY_T1, CANNY_T2, EDGE_EMA_ALPHA,
    Q_STEP, PIN_CRIT, PIN_LOW, PIN_MID
)

LIGHT_LEVELS = ["BRIGHT", "NORMAL", "DIM", "DARK"]

def update_ema(prev, new, a):
    if prev is None:
        return new
    return (1.0 - a) * prev + a * new

def compute_light_metrics(frame):
    small = cv2.resize(frame, (ANALYZE_W, ANALYZE_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean_y = float(gray.mean())
    dark_ratio = float(np.mean(gray < DARK_PIXEL_Y))
    return mean_y, dark_ratio

def classify_light(mean_y, dark_ratio):
    if mean_y >= MEAN_BINS[0] and dark_ratio <= DARK_BINS[0]:
        return 0
    if mean_y >= MEAN_BINS[1] and dark_ratio <= DARK_BINS[1]:
        return 1
    if mean_y >= MEAN_BINS[2] and dark_ratio <= DARK_BINS[2]:
        return 2
    return 3

def stable_level_switch(active_level, detected_level, candidate_level, candidate_since, now, hold_sec):
    if active_level is None:
        return detected_level, None, None
    if detected_level == active_level:
        return active_level, None, None
    if candidate_level != detected_level:
        return active_level, detected_level, now
    if candidate_since is not None and (now - candidate_since) >= hold_sec:
        return detected_level, None, None
    return active_level, candidate_level, candidate_since

def compute_edge_density(frame):
    small = cv2.resize(frame, (EDGE_W, EDGE_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, CANNY_T1, CANNY_T2)
    return float(np.count_nonzero(edges)) / float(edges.size)

def quantize_percent01(x01, step=5):
    p = x01 * 100.0
    q = int(p / step) * step
    if q < 0:
        q = 0
    if q > 100:
        q = 100
    return int(q)

def pin_bin_from_pct(pct):
    if pct is None:
        return "NA"
    p = float(pct)
    if p < PIN_CRIT:
        return "CRIT"
    if p < PIN_LOW:
        return "LOW"
    if p < PIN_MID:
        return "MID"
    return "HIGH"

def make_context_id(light_name, dark_q, edge_q, pin_bin):
    return "%s__dark%03d__edge%03d__%s" % (light_name, int(dark_q), int(edge_q), str(pin_bin))

class ContextTracker:
    def __init__(self):
        self.ema_mean = None
        self.ema_dark = None
        self.ema_edge = None

        self.active_level = None
        self.candidate_level = None
        self.candidate_since = None

        self.last_ctx_id = None
        self.ctx_since = None

    def update(self, frame, now_ts, pin_pct):
        mean_y, dark_ratio = compute_light_metrics(frame)
        self.ema_mean = update_ema(self.ema_mean, mean_y, EMA_ALPHA)
        self.ema_dark = update_ema(self.ema_dark, dark_ratio, EMA_ALPHA)

        det_level = classify_light(self.ema_mean, self.ema_dark)
        self.active_level, self.candidate_level, self.candidate_since = stable_level_switch(
            self.active_level, det_level, self.candidate_level, self.candidate_since, now_ts, STABLE_HOLD_SEC
        )
        light_name = LIGHT_LEVELS[int(self.active_level) if self.active_level is not None else 0]

        edge = compute_edge_density(frame)
        self.ema_edge = update_ema(self.ema_edge, edge, EDGE_EMA_ALPHA)

        dark_q = quantize_percent01(float(self.ema_dark), Q_STEP)
        edge_q = quantize_percent01(float(self.ema_edge if self.ema_edge is not None else 0.0), Q_STEP)
        pin_bin = pin_bin_from_pct(pin_pct)

        cid = make_context_id(light_name, dark_q, edge_q, pin_bin)
        if cid != self.last_ctx_id:
            self.last_ctx_id = cid
            self.ctx_since = now_ts

        stable = (self.ctx_since is not None) and ((now_ts - self.ctx_since) >= 1.0)

        ctx_obj = {
            "light": light_name,
            "dark_q": int(dark_q),
            "edge_q": int(edge_q),
            "pin_bin": str(pin_bin),
        }
        return cid, ctx_obj, stable, light_name, dark_q, edge_q, pin_bin
