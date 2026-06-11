# adaptive_res.py
# ASCII-only.

import time
import statistics
import cv2


def resize_keep_aspect(frame, target_w):
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    target_w = int(max(1, min(w, int(target_w))))
    if target_w == w:
        return frame
    target_h = int(h * (float(target_w) / float(w)))
    if target_h < 1:
        target_h = 1
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_canvas(base_h, base_w, small_bgr):
    canvas = cv2.resize(small_bgr[0:1, 0:1], (base_w, base_h))
    canvas[:] = 0

    sh, sw = small_bgr.shape[:2]
    y0 = (base_h - sh) // 2
    x0 = (base_w - sw) // 2
    if y0 < 0 or x0 < 0:
        return cv2.resize(small_bgr, (base_w, base_h), interpolation=cv2.INTER_AREA)

    canvas[y0:y0 + sh, x0:x0 + sw] = small_bgr
    return canvas


def _has_object_from_result(r0, conf_th):
    if r0.boxes is None:
        return False, 0.0, 0
    confs = []
    try:
        for b in r0.boxes:
            c = float(b.conf)
            if c >= float(conf_th):
                confs.append(c)
    except Exception:
        return False, 0.0, 0
    if not confs:
        return False, 0.0, 0
    return True, float(max(confs)), int(len(confs))


def score_width_on_frame(model, frame_bgr, test_w,
                         test_frames, conf_th, min_det_rate):

    det_flags = []
    conf_list = []

    for _ in range(int(test_frames)):
        fr_rs = resize_keep_aspect(frame_bgr, test_w)
        try:
            results = model(fr_rs, verbose=False)
            r0 = results[0]
        except Exception:
            continue

        has_obj, conf_max, _n = _has_object_from_result(r0, conf_th)
        det_flags.append(1 if has_obj else 0)
        if has_obj:
            conf_list.append(float(conf_max))

    if not det_flags:
        return False, 0.0, 0.0

    det_rate = float(sum(det_flags)) / float(len(det_flags))
    med_conf = float(statistics.median(conf_list)) if conf_list else 0.0
    is_good = (det_rate >= float(min_det_rate))
    return is_good, det_rate, med_conf


def auto_find_min_good_width_on_frame(
    model,
    frame_bgr,
    base_w,
    shrink_ratio=0.85,
    min_w=96,
    test_frames=10,
    min_det_rate=0.30,
    binary_iters=8,
    conf_th=0.25,
):
    base_w = int(base_w)
    cur = int(base_w)
    last_good = cur
    first_bad = None

    while True:
        is_good, _, _ = score_width_on_frame(
            model, frame_bgr, cur,
            test_frames, conf_th, min_det_rate
        )

        if is_good:
            last_good = cur
            next_w = int(cur * float(shrink_ratio))
            if next_w >= cur:
                next_w = cur - 1
            if next_w < int(min_w):
                return int(last_good)
            cur = next_w
        else:
            first_bad = cur
            break

    lo = int(first_bad)
    hi = int(last_good)
    if lo > hi:
        lo, hi = hi, lo

    for _ in range(int(binary_iters)):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2

        is_good, _, _ = score_width_on_frame(
            model, frame_bgr, mid,
            test_frames, conf_th, min_det_rate
        )

        if is_good:
            hi = mid
        else:
            lo = mid

    return int(hi)


# ============================================================
# VISUAL CALIBRATION (like adaptive_res_autolimit_visual_boxes.py)
# Works with "shared latest frame" via a callback get_frame().
# ============================================================

def _draw_overlay(img, lines, top=25):
    y = int(top)
    for s in lines:
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 28


def yolo_predict_and_annotate(model, frame_bgr, conf_th):
    # Returns: annotated_bgr, has_obj, conf_max, n_boxes
    results = None
    try:
        results = model.predict(source=frame_bgr, verbose=False)
    except Exception:
        results = None

    if not results:
        return frame_bgr, False, 0.0, 0

    r0 = results[0]
    annotated = frame_bgr
    try:
        annotated = r0.plot()
    except Exception:
        annotated = frame_bgr

    has_obj, conf_max, n_boxes = _has_object_from_result(r0, conf_th)
    return annotated, bool(has_obj), float(conf_max), int(n_boxes)


def score_resolution_visual_shared(model, get_frame, base_w, base_h, test_w,
                                   win_name, phase_label,
                                   test_frames=40,
                                   min_det_rate=0.30,
                                   conf_th=0.25,
                                   warmup_frames=3):
    test_w = int(max(1, min(int(base_w), int(test_w))))

    det_flags = []
    conf_list = []

    # warmup (like sample)
    for _ in range(int(warmup_frames)):
        fr = get_frame()
        if fr is None:
            time.sleep(0.01)
            continue
        fr_rs = resize_keep_aspect(fr, test_w)
        yolo_predict_and_annotate(model, fr_rs, conf_th)

    for i in range(int(test_frames)):
        fr = get_frame()
        if fr is None:
            time.sleep(0.01)
            continue

        fr_rs = resize_keep_aspect(fr, test_w)
        annotated_rs, has_obj, conf_max, n_boxes = yolo_predict_and_annotate(model, fr_rs, conf_th)

        det_flags.append(1 if has_obj else 0)
        if has_obj:
            conf_list.append(float(conf_max))

        det_rate = float(sum(det_flags)) / float(len(det_flags))
        med_conf = float(statistics.median(conf_list)) if conf_list else 0.0
        is_good_now = (det_rate >= float(min_det_rate))

        canvas = make_canvas(base_h, base_w, annotated_rs)

        lines = [
            "CALIBRATING: %s" % str(phase_label),
            "TEST w=%d  frame=%d/%d" % (int(test_w), int(i + 1), int(test_frames)),
            "det_rate=%.2f need>=%.2f  status=%s" % (float(det_rate), float(min_det_rate), "GOOD" if is_good_now else "BAD"),
            "has_obj=%s n=%d conf=%.2f med_conf=%.2f" % (str(has_obj), int(n_boxes), float(conf_max), float(med_conf)),
            "Press q/ESC to abort",
        ]
        _draw_overlay(canvas, lines, top=25)

        cv2.imshow(win_name, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            return False, det_rate, med_conf, True

    if not det_flags:
        return False, 0.0, 0.0, False

    det_rate = float(sum(det_flags)) / float(len(det_flags))
    med_conf = float(statistics.median(conf_list)) if conf_list else 0.0
    is_good = (det_rate >= float(min_det_rate))
    return bool(is_good), float(det_rate), float(med_conf), False


def auto_find_min_good_width_visual_shared(model, get_frame, base_w, base_h,
                                           win_name="Realtime",
                                           test_frames=40,
                                           min_det_rate=0.30,
                                           shrink_ratio=0.85,
                                           min_w=96,
                                           binary_iters=10,
                                           conf_th=0.25):
    cur = int(base_w)
    last_good = int(cur)
    first_bad = None

    # progressive shrink
    while True:
        is_good, det_rate, med_conf, aborted = score_resolution_visual_shared(
            model, get_frame, base_w, base_h, cur,
            win_name=win_name,
            phase_label="shrink",
            test_frames=test_frames,
            min_det_rate=min_det_rate,
            conf_th=conf_th
        )
        if aborted:
            return int(last_good)

        if is_good:
            last_good = int(cur)
            next_w = int(cur * float(shrink_ratio))
            if next_w >= cur:
                next_w = cur - 1
            if next_w < int(min_w):
                return int(last_good)
            cur = int(next_w)
        else:
            first_bad = int(cur)
            break

    # binary refine
    lo = int(first_bad)
    hi = int(last_good)
    if lo > hi:
        lo, hi = hi, lo

    for _ in range(int(binary_iters)):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        is_good, det_rate, med_conf, aborted = score_resolution_visual_shared(
            model, get_frame, base_w, base_h, mid,
            win_name=win_name,
            phase_label="refine",
            test_frames=test_frames,
            min_det_rate=min_det_rate,
            conf_th=conf_th
        )
        if aborted:
            return int(hi)
        if is_good:
            hi = int(mid)
        else:
            lo = int(mid)

    return int(hi)
