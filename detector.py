"""
detector.py
------------
The simple, original approach: per-ROI Z-score anomaly detection.

For each circular ROI (a camera lens/module) the pixels inside are one
population. A pixel that is brighter than the population mean by more
than `sigma` standard deviations is a candidate defect. That's it - no
texture channels, no radial models, none of that.

Two deliberate choices matched to what we actually inspect:

  * ONE-SIDED (bright only). Dust and cloth strings are white/light on
    the module surface, so only brighter-than-surroundings spots count.
    Darker patches (shadows, coating dark spots) are ignored.

  * WHITE ONLY. Real dust/string is white/gray = low colour saturation.
    Anything meaningfully coloured (blue/green AR-coating reflections,
    yellow glare) is dropped, however bright it looks in grayscale.
    This is a single saturation threshold - simple and predictable.

Thin strings are preserved by NOT using a morphological "open" (which
erodes 1-2px lines). Instead the bright mask is closed to bridge small
gaps, then connected components are filtered by pixel count, so a long
thin string survives while isolated sensor-noise specks are dropped.

Blobs are labelled dust (compact) or string (elongated) purely for the
on-image colour; both count as a defect either way.
"""

import cv2
import numpy as np

DUST_COLOR = (0, 0, 255)      # red   (BGR)
STRING_COLOR = (0, 210, 255)  # amber (BGR)

DEFAULT_PARAMS = dict(
    sigma=3.5,                # brightness Z-score threshold
    min_area=6,               # min blob area in pixels (auto-scaled to resolution by the app)
    white_only=True,          # ignore coloured pixels, keep only white/gray defects
    white_max_saturation=60,  # HSV S above this = "coloured", not a defect
    blur_ksize=3,             # pre-blur to calm pixel noise (odd, small)
    string_elongation=3.0,    # length/width above which a blob is called a "string"
    edge_margin_px=4,         # shrink each ROI by this so the lens rim doesn't fire
)


class Result:
    def __init__(self):
        self.defects = []      # [{x,y,r,area,cls,roi_index}]
        self.verdict = "OK"
        self.annotated = None
        self.counts = {"dust": 0, "string": 0}

    def summary(self):
        if not self.defects:
            return "clean"
        bits = []
        if self.counts["dust"]:
            bits.append(f"{self.counts['dust']} dust")
        if self.counts["string"]:
            bits.append(f"{self.counts['string']} string")
        return ", ".join(bits)


def _circular_mask(shape_hw, cx, cy, r):
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(int(round(r)), 1), 255, -1)
    return m


def _detect_in_roi(gray, sat, cx, cy, radius, p):
    h, w = gray.shape
    mask = _circular_mask((h, w), cx, cy, radius)
    inside = mask > 0
    pix = gray[inside]
    if pix.size < 50:
        return []

    mean = float(np.mean(pix))
    std = float(np.std(pix))
    if std < 1e-6:
        return []

    z = (gray.astype(np.float32) - mean) / std
    bright = (z > p["sigma"]) & inside            # one-sided: white/bright only

    if p["white_only"] and sat is not None:
        bright &= (sat <= p["white_max_saturation"])   # drop coloured pixels

    bright_u8 = bright.astype(np.uint8) * 255

    # Close to bridge a thin string's small gaps; do NOT open (that would
    # erase the string). Isolated noise specks are removed by the area
    # filter below instead.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_u8, connectivity=8)
    blobs = []
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])   # true pixel count (honest for thin shapes)
        if area < p["min_area"]:
            continue
        ys, xs = np.where(labels == lbl)
        pts = np.column_stack([xs, ys]).astype(np.int32)
        (_, (rw, rh), _) = cv2.minAreaRect(pts)
        elong = max(rw, rh) / max(min(rw, rh), 1.0)
        cls = "string" if elong >= p["string_elongation"] else "dust"
        (bx, by), br = cv2.minEnclosingCircle(pts)
        blobs.append({"x": float(bx), "y": float(by), "r": max(float(br), 3.0),
                      "area": float(area), "cls": cls, "elongation": float(elong)})
    return blobs


def inspect(frame, rois, params=None):
    """
    frame: BGR (H,W,3) or grayscale (H,W)
    rois:  [{"cx","cy","r"}, ...] in full-frame pixel coords
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        annotated = frame.copy()
    else:
        gray = frame
        sat = None
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    kb = int(p["blur_ksize"])
    if kb >= 3:
        if kb % 2 == 0:
            kb += 1
        gray = cv2.GaussianBlur(gray, (kb, kb), 0)

    res = Result()
    for i, roi in enumerate(rois):
        r_an = max(roi["r"] - p["edge_margin_px"], 4)
        for b in _detect_in_roi(gray, sat, roi["cx"], roi["cy"], r_an, p):
            b["roi_index"] = i
            res.defects.append(b)
            res.counts[b["cls"]] += 1
        cv2.circle(annotated, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]), (90, 90, 96), 1)

    for b in res.defects:
        color = STRING_COLOR if b["cls"] == "string" else DUST_COLOR
        cv2.circle(annotated, (int(b["x"]), int(b["y"])), int(b["r"]) + 5, color, 2)

    res.verdict = "NG" if res.defects else "OK"
    res.annotated = annotated
    return res
