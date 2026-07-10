"""
detector.py
------------
Per-ROI Z-score anomaly detection for white dust and thin cloth strings,
with PHYSICAL SIZE measurement so the accept/reject rule matches the
actual spec instead of an arbitrary pixel count.

Detection (Updated with Local Background Subtraction):
  * local high-pass filter - flattens uneven lighting/vignetting so isolated
    small dust blobs don't get lost in the global background mean.
  * one-sided Z-score - only pixels BRIGHTER than the local ROI mean count,
    because dust/string/glue are white on the module surface.
  * white-only gate - coloured pixels (blue/green AR-coating reflections,
    yellow glare) are dropped by an HSV saturation check.
  * no morphological "open" (it erases 1-2px strings); tiny components
    are removed by a pixel-count noise floor instead.

Sizing:
  Each blob's size is its MAX FERET LENGTH - the largest end-to-end
  distance across the blob, computed exactly as the maximum pairwise
  distance between its convex-hull vertices.

Calibration:
  mm_per_px cannot be derived from the image alone. One physical reference is
  required. If mm_per_px is 0 the detector stays in pixel mode.
"""

import cv2
import numpy as np

DUST_COLOR = (0, 0, 255)      # red   (BGR)
STRING_COLOR = (0, 210, 255)  # amber (BGR)

DEFAULT_PARAMS = dict(
    sigma=3.5,                # brightness Z-score threshold
    white_only=True,          # ignore coloured pixels
    white_max_saturation=60,  # HSV S above this = "coloured", not a defect
    blur_ksize=3,             # pre-blur to calm sensor noise (odd)
    string_elongation=3.0,    # length/width above which a blob is a "string"
    edge_margin_px=4,         # shrink each ROI so the lens rim doesn't fire
    min_length_mm=0.1,        # SPEC: ignore anything whose max span is under this
    mm_per_px=0.0,            # 0 = uncalibrated (pixel mode)
    min_pixels=4,             # hard noise floor, always applied
    size_bias_px=0.0,         # subtracted from every measured span
)


class Result:
    def __init__(self):
        self.defects = []
        self.verdict = "OK"
        self.annotated = None
        self.counts = {"dust": 0, "string": 0}
        self.calibrated = False

    def largest(self):
        if not self.defects:
            return None
        return max(self.defects, key=lambda b: b["feret_px"])

    def summary(self):
        if not self.defects:
            return "clean"
        bits = []
        if self.counts["dust"]:
            bits.append(f"{self.counts['dust']} dust")
        if self.counts["string"]:
            bits.append(f"{self.counts['string']} string")
        big = self.largest()
        size = (f"{big['feret_mm']:.3f} mm" if self.calibrated else f"{big['feret_px']:.0f} px")
        return f"{', '.join(bits)}  \u2022  largest {size}"


def _circular_mask(shape_hw, cx, cy, r):
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(int(round(r)), 1), 255, -1)
    return m


def _max_feret_px(pts):
    """Largest end-to-end distance across the blob, in pixels."""
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    if len(hull) < 2:
        return 1.0
    d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max())) + 1.0


def _detect_in_roi(gray, gray_raw, sat, cx, cy, radius, p):
    h, w = gray.shape
    mask = _circular_mask((h, w), cx, cy, radius)
    inside = mask > 0
    pix = gray[inside]
    if pix.size < 50:
        return []

    # --- Local Background Subtraction (High-Pass) ---
    bg_ksize = int(max(radius // 4, 31))
    if bg_ksize % 2 == 0: bg_ksize += 1

    bg = cv2.GaussianBlur(gray, (bg_ksize, bg_ksize), 0)
    high_pass = gray.astype(np.float32) - bg.astype(np.float32)
    hp_pix = high_pass[inside]

    mean = float(np.mean(hp_pix))
    std = float(np.std(hp_pix))
    
    if std < 1e-6:
        return []

    z = (high_pass - mean) / std
    bright = (z > p["sigma"]) & inside

    if p["white_only"] and sat is not None:
        bright &= (sat <= p["white_max_saturation"])

    bright_u8 = bright.astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_CLOSE, k)

    # Sizing mask (exact extent from the raw image)
    raw_bg = cv2.GaussianBlur(gray_raw, (bg_ksize, bg_ksize), 0)
    hp_raw = gray_raw.astype(np.float32) - raw_bg.astype(np.float32)
    hp_raw_pix = hp_raw[inside]
    raw_mean = float(np.mean(hp_raw_pix))
    raw_std = float(np.std(hp_raw_pix))

    if raw_std > 1e-6:
        z_raw = (hp_raw - raw_mean) / raw_std
        sharp = (z_raw > p["sigma"]) & inside
        if p["white_only"] and sat is not None:
            sharp &= (sat <= p["white_max_saturation"])
    else:
        sharp = bright

    mm_per_px = float(p["mm_per_px"])
    calibrated = mm_per_px > 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_u8, connectivity=8)
    blobs = []
    for lbl in range(1, n):
        area_px = int(stats[lbl, cv2.CC_STAT_AREA])
        if area_px < p["min_pixels"]:
            continue
        blob_sel = labels == lbl

        sharp_sel = blob_sel & sharp
        if int(np.count_nonzero(sharp_sel)) >= 2:
            mys, mxs = np.where(sharp_sel)
            m_area = int(sharp_sel.sum())
        else:
            mys, mxs = np.where(blob_sel)
            m_area = area_px
        m_pts = np.column_stack([mxs, mys]).astype(np.int32)

        feret_px = max(_max_feret_px(m_pts) - float(p["size_bias_px"]), 0.1)
        feret_mm = feret_px * mm_per_px if calibrated else None
        area_mm2 = m_area * (mm_per_px ** 2) if calibrated else None

        if calibrated and feret_mm < p["min_length_mm"]:
            continue

        (_, (rw, rh), _) = cv2.minAreaRect(m_pts)
        elong = max(rw, rh) / max(min(rw, rh), 1.0)
        cls = "string" if elong >= p["string_elongation"] else "dust"
        (bx, by), br = cv2.minEnclosingCircle(m_pts)
        blobs.append({
            "x": float(bx), "y": float(by), "r": max(float(br), 3.0),
            "area_px": float(m_area), "feret_px": float(feret_px),
            "feret_mm": feret_mm, "area_mm2": area_mm2,
            "cls": cls, "elongation": float(elong),
        })
    return blobs


def inspect(frame, rois, params=None):
    """
    frame: BGR (H,W,3) or grayscale (H,W)
    rois:  [{"cx","cy","r"}, ...] in FULL-RESOLUTION pixel coords
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

    gray_raw = gray
    kb = int(p["blur_ksize"])
    if kb >= 3:
        if kb % 2 == 0:
            kb += 1
        gray = cv2.GaussianBlur(gray, (kb, kb), 0)

    res = Result()
    res.calibrated = float(p["mm_per_px"]) > 0

    h, w = annotated.shape[:2]
    th = max(1, int(round(max(h, w) / 1600)))
    fs = max(0.4, min(1.4, max(h, w) / 3600.0))

    for i, roi in enumerate(rois):
        r_an = max(roi["r"] - p["edge_margin_px"], 4)
        for b in _detect_in_roi(gray, gray_raw, sat, roi["cx"], roi["cy"], r_an, p):
            b["roi_index"] = i
            res.defects.append(b)
            res.counts[b["cls"]] += 1
        cv2.circle(annotated, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]),
                   (90, 90, 96), th)

    for b in res.defects:
        color = STRING_COLOR if b["cls"] == "string" else DUST_COLOR
        c = (int(b["x"]), int(b["y"]))
        rr = int(b["r"]) + 6 * th
        cv2.circle(annotated, c, rr, color, th + 1)
        label = f"{b['feret_mm']:.3f}mm" if res.calibrated else f"{b['feret_px']:.0f}px"
        cv2.putText(annotated, label, (c[0] + rr + 4, c[1] - rr),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)

    res.verdict = "NG" if res.defects else "OK"
    res.annotated = annotated
    return res


def mm_per_px_from_roi(roi_radius_px, real_diameter_mm):
    """Calibrate from a drawn ROI whose real diameter is known."""
    if roi_radius_px <= 0 or real_diameter_mm <= 0:
        return 0.0
    return float(real_diameter_mm) / (2.0 * float(roi_radius_px))


def px_for_length(length_mm, mm_per_px):
    """How many pixels a given physical length spans (for the UI readout)."""
    if mm_per_px <= 0:
        return 0.0
    return float(length_mm) / float(mm_per_px)
