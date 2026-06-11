# quality_metrics.py
# ASCII-only.

import time
import cv2
import numpy as np

from config import (
    CHG_W, CHG_H, STABLE_CHG_THRESH, DELTA_CONF_THRESH,
    IOU_MATCH_THRESH, DISAPPEAR_K, COOLDOWN_MS, EXPAND_RATIO,
    DIFF_THRESH, PATCH_SIZE
)

def frame_change_score(prev_small, frame_bgr):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (CHG_W, CHG_H), interpolation=cv2.INTER_AREA)
    if prev_small is None:
        return g, 0.0
    d = cv2.absdiff(g, prev_small)
    return g, float(np.mean(d)) / 255.0

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def expand_box(x1, y1, x2, y2, w, h, r):
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nw = bw * (1.0 + 2.0 * r)
    nh = bh * (1.0 + 2.0 * r)
    ex1 = cx - nw * 0.5
    ey1 = cy - nh * 0.5
    ex2 = cx + nw * 0.5
    ey2 = cy + nh * 0.5
    return clip_box(ex1, ey1, ex2, ey2, w, h)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom

def normalize01(a):
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)

def preprocess_feat(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (5, 5), 1.0)

    gf = g.astype(np.float32) / 255.0
    gf = gf - float(np.mean(gf))

    low = cv2.blur(gf, (17, 17))
    low = normalize01(low)

    g01 = gf - float(np.min(gf))
    mx = float(np.max(g01))
    if mx > 1e-6:
        g01 = g01 / mx
    g8 = (g01 * 255.0).astype(np.uint8)
    edge = cv2.Canny(g8, 60, 120).astype(np.float32) / 255.0

    return 0.6 * low + 0.4 * edge

def diff_score(f1, f2):
    return float(np.mean(np.abs(f1 - f2)))

def get_dets(results0):
    b = results0.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    out = []
    for i in range(len(xyxy)):
        out.append((xyxy[i], float(conf[i])))
    return out

class MistConfMeter:
    def __init__(self):
        # conf
        self.prev_small = None
        self.prev_conf_valid = False
        self.prev_conf = 0.0
        self.stable_samples = 0
        self.event_count = 0

        # mistake (single dominant track)
        self.has_track = False
        self.last_box = None
        self.last_feat = None
        self.missing = 0
        self.last_drop_ts = 0.0

        self.calls = 0
        self.mistakes = 0

    def update(self, frame_bgr, results0, now_ts):
        self.calls += 1

        dets = get_dets(results0)
        seen = (len(dets) > 0)

        # frame change for conf gate
        self.prev_small, chg = frame_change_score(self.prev_small, frame_bgr)

        # conf event-based
        if (chg < STABLE_CHG_THRESH) and seen:
            max_conf = 0.0
            best = None
            for (xyxy, c) in dets:
                if c > max_conf:
                    max_conf = c
                    best = xyxy
            self.stable_samples += 1
            if self.prev_conf_valid:
                if abs(max_conf - self.prev_conf) > DELTA_CONF_THRESH:
                    self.event_count += 1
            self.prev_conf = max_conf
            self.prev_conf_valid = True

        # mistake simplified: keep best box by conf, track by IoU
        h, w = frame_bgr.shape[:2]

        if not self.has_track:
            if seen:
                # init track = max conf
                best_conf = -1.0
                best_xyxy = None
                for (xyxy, c) in dets:
                    if c > best_conf:
                        best_conf = c
                        best_xyxy = xyxy
                x1, y1, x2, y2 = map(int, best_xyxy)
                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
                ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_RATIO)
                roi = frame_bgr[ey1:ey2, ex1:ex2]
                if roi.size > 0:
                    self.last_feat = preprocess_feat(roi)
                    self.last_box = (ex1, ey1, ex2, ey2)
                    self.has_track = True
                    self.missing = 0
            return

        # has track
        if seen:
            # find det that matches track
            best_iou = -1.0
            best_xyxy = None
            for (xyxy, c) in dets:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
                ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_RATIO)
                iou = iou_xyxy(self.last_box, (ex1, ey1, ex2, ey2))
                if iou > best_iou:
                    best_iou = iou
                    best_xyxy = (ex1, ey1, ex2, ey2)
            if best_iou >= IOU_MATCH_THRESH and best_xyxy is not None:
                ex1, ey1, ex2, ey2 = best_xyxy
                roi = frame_bgr[ey1:ey2, ex1:ex2]
                if roi.size > 0:
                    self.last_feat = preprocess_feat(roi)
                    self.last_box = best_xyxy
                self.missing = 0
            else:
                self.missing += 1
        else:
            self.missing += 1

        if self.missing >= DISAPPEAR_K:
            if (now_ts - self.last_drop_ts) * 1000.0 >= COOLDOWN_MS:
                ex1, ey1, ex2, ey2 = self.last_box
                roi_now = frame_bgr[ey1:ey2, ex1:ex2]
                if roi_now.size > 0 and self.last_feat is not None:
                    feat_now = preprocess_feat(roi_now)
                    sc = diff_score(self.last_feat, feat_now)
                    if sc < DIFF_THRESH:
                        self.mistakes += 1
                self.last_drop_ts = now_ts
            self.missing = 0

    def get_rates(self):
        mist_rate = float(self.mistakes) / float(max(1, self.calls))
        conf_event_rate = float(self.event_count) / float(max(1, self.stable_samples))
        return mist_rate, conf_event_rate
