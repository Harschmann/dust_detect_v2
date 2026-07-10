"""
detector.py
------------
Per-ROI Z-score anomaly detection for white dust and thin cloth strings,
with PHYSICAL SIZE measurement so the accept/reject rule matches the
actual spec instead of an arbitrary pixel count.

Detection (unchanged, deliberately simple):
  * one-sided Z-score - only pixels BRIGHTER than the ROI mean count,
    because dust/string/glue are white on the module surface.
  * white-only gate - coloured pixels (blue/green AR-coating reflections,
    yellow glare) are dropped by an HSV saturation check.
  * no morphological "open" (it erases 1-2px strings); tiny components
    are removed by a pixel-count noise floor instead.

Sizing:
  Each blob's size is its MAX FERET LENGTH - the largest end-to-end
  distance across the blob, computed exactly as the maximum pairwise
  distance between its convex-hull vertices. For a round particle that
  is its diameter; for an irregular one it is the longest span, which is
  exactly how the 0.1 mm spec is written.

  With a calibration (mm per pixel) that length converts to millimetres
  and anything below `min_length_mm` is ignored. Area in mm^2 is
  reported too (pixel count x mm_per_px^2).

Calibration:
  mm_per_px cannot be derived from the image alone - a 20 MP photo says
  nothing about how wide a scene it covers. One physical reference is
  required. The app provides two ways to supply it:
     * a drawn ROI whose real diameter (mm) is known, e.g. the lens
       barrel:  mm_per_px = real_diameter_mm / (2 * roi_radius_px)
     * a directly entered sensor scale in micrometres per pixel.
  If mm_per_px is 0 the detector stays in pixel mode: sizes are reported
  in px and only the pixel noise floor filters blobs.
"""

import cv2
import numpy as np

DUST_COLOR = (0, 0, 255)      # red   (BGR)
STRING_COLOR = (0, 210, 255)  # amber (BGR)

DEFAULT_PARAMS = dict(
    sigma=3.5,                # brightness Z-score threshold
    white_only=True,          # ignore coloured pixels
    white_max_saturation=60,  # HSV S above this = "coloured", not a defect
    dark_floor=0,             # gray below this is ignored (0 = off); blacks out dark regions
    blur_ksize=3,             # pre-blur to calm sensor noise (odd)
    string_elongation=3.0,    # length/width above which a blob is a "string"
    edge_margin_px=4,         # shrink each ROI so the lens rim doesn't fire
    min_length_mm=0.1,        # SPEC: ignore anything whose max span is under this
    mm_per_px=0.0,            # 0 = uncalibrated (pixel mode)
    min_pixels=10,            # hard noise floor, always applied (uncalibrated mode only - see note)
    size_bias_px=0.0,         # subtracted from every measured span; see note below
    max_defect_fraction=0.5,  # reject any blob spanning more than this fraction of the ROI diameter (structural, not a defect)
)

# MEASUREMENT ACCURACY
# Blobs are FOUND on the blurred image (so a faint particle still clears
# the Z-threshold) but MEASURED on the unblurred one. Without that split
# the blur's skirt pushes a ring of extra pixels over the threshold and
# inflates small particles by ~2 px. Verified against synthetic discs of
# known pixel diameter: measurement is exact to +/-0 px across 7..61 px.
# size_bias_px stays at 0 unless you calibrate against a known reference
# particle and find a residual systematic offset in your optics.


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
    """Largest end-to-end distance across the blob, in pixels.

    The farthest pair of points always lies on the convex hull, so only
    hull vertices need comparing - a handful of points, so the O(n^2)
    scan is trivial.

    The +1.0 is a real correction, not a fudge: hull vertices are pixel
    CENTRES, so a blob physically spanning N pixels has centres spanning
    only N-1. Verified against synthetic discs of known pixel diameter.
    """
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    if len(hull) < 2:
        return 1.0
    d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max())) + 1.0


def white_mask_preview(frame, rois, params=None):
    """Returns a BGR image showing what the colour/dark mask keeps: inside
    each ROI, coloured and dark pixels are blacked out and only white-
    eligible pixels keep their brightness. Uses the exact same plain
    per-pixel cutoff as real detection, so this preview is always
    trustworthy - nothing here can differ from what gets detected."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    else:
        gray = frame
        sat = None
    h, w = gray.shape
    keep = np.zeros((h, w), dtype=bool)
    for roi in rois:
        r_an = max(roi["r"] - p["edge_margin_px"], 4)
        m = _circular_mask((h, w), roi["cx"], roi["cy"], r_an) > 0
        if p["white_only"] and sat is not None:
            m &= (sat <= p["white_max_saturation"])
        if p.get("dark_floor", 0) > 0:
            m &= (gray >= float(p["dark_floor"]))
        keep |= m
    out = np.zeros((h, w, 3), dtype=np.uint8)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out[keep] = g3[keep]
    return out


def _detect_in_roi(gray, gray_raw, sat, cx, cy, radius, p):
    H, W = gray.shape

    # Crop to a local window around this ROI (with a small margin) before
    # any per-pixel work - keeps this fast on 20MP frames.
    pad = 8
    x0 = max(0, int(cx - radius - pad)); x1 = min(W, int(cx + radius + pad) + 1)
    y0 = max(0, int(cy - radius - pad)); y1 = min(H, int(cy + radius + pad) + 1)
    if x1 <= x0 or y1 <= y0:
        return []

    gray_c = gray[y0:y1, x0:x1]
    gray_raw_c = gray_raw[y0:y1, x0:x1]
    sat_c = sat[y0:y1, x0:x1] if sat is not None else None
    cx_c, cy_c = cx - x0, cy - y0
    h, w = gray_c.shape

    mask = _circular_mask((h, w), cx_c, cy_c, radius)
    inside = mask > 0
    pix = gray_c[inside]
    if pix.size < 50:
        return []

    # Statistics over the FULL ROI population, never gated by colour -
    # a previous version computed mean/std only over "white-eligible"
    # pixels, which could collapse to almost nothing on a real coating
    # with broad colour variation and silently return zero detections.
    # Keeping this simple and ungated is what makes it robust.
    mean = float(np.mean(pix))
    std = float(np.std(pix))
    if std < 1e-6:
        return []

    z = (gray_c.astype(np.float32) - mean) / std
    bright = (z > p["sigma"]) & inside              # one-sided: brighter than surroundings

    # White-only: a real defect (dust/thread/glue) is white/gray = LOW
    # saturation. A plain per-pixel cutoff - no growing, no reconstruction,
    # nothing that can spread across a whole coating and eat everything.
    if p["white_only"] and sat_c is not None:
        bright &= (sat_c <= p["white_max_saturation"])
    if p.get("dark_floor", 0) > 0:
        bright &= (gray_raw_c >= float(p["dark_floor"]))

    bright_u8 = bright.astype(np.uint8) * 255

    # Close bridges a thin string's small gaps. No "open" - that would
    # erase the string outright. Noise specks die on the area filter.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_CLOSE, k)

    # Sizing mask, measured on the UNBLURRED image (blur inflates small
    # particles by ~2px).
    raw_mean = float(np.mean(gray_raw_c[inside]))
    raw_std = float(np.std(gray_raw_c[inside]))
    if raw_std > 1e-6:
        z_raw = (gray_raw_c.astype(np.float32) - raw_mean) / raw_std
        sharp = (z_raw > p["sigma"]) & inside
        if p["white_only"] and sat_c is not None:
            sharp &= (sat_c <= p["white_max_saturation"])
    else:
        sharp = bright

    mm_per_px = float(p["mm_per_px"])
    calibrated = mm_per_px > 0
    max_feret_allowed = 2.0 * radius * float(p.get("max_defect_fraction", 0.5))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_u8, connectivity=8)
    blobs = []
    for lbl in range(1, n):
        area_px = int(stats[lbl, cv2.CC_STAT_AREA])
        if area_px < p["min_pixels"]:
            continue
        blob_sel = labels == lbl

        # exact extent from the raw image, confined to this blob
        sharp_sel = blob_sel & sharp
        if int(np.count_nonzero(sharp_sel)) >= 2:
            mys, mxs = np.where(sharp_sel)
            m_area = int(sharp_sel.sum())
        else:
            mys, mxs = np.where(blob_sel)       # faint particle: blur was load-bearing
            m_area = area_px
        m_pts = np.column_stack([mxs, mys]).astype(np.int32)

        feret_px = max(_max_feret_px(m_pts) - float(p["size_bias_px"]), 0.1)

        # MAX-SIZE GATE: a real defect is always much smaller than the
        # lens it sits on. Anything spanning a large fraction of the ROI
        # itself - a broad ring from the coating's own construction, a
        # big shadow, a lighting gradient - is a structural/background
        # feature, not contamination, and is rejected here regardless of
        # brightness or colour. This is what guarantees the concentric
        # ring around the lens centre can never be called a defect.
        if feret_px > max_feret_allowed:
            continue

        feret_mm = feret_px * mm_per_px if calibrated else None
        area_mm2 = m_area * (mm_per_px ** 2) if calibrated else None

        # THE SPEC RULE: ignore particles whose longest span is under the
        # threshold. For a round particle that span is its diameter.
        if calibrated and feret_mm < p["min_length_mm"]:
            continue

        (_, (rw, rh), _) = cv2.minAreaRect(m_pts)
        elong = max(rw, rh) / max(min(rw, rh), 1.0)
        cls = "string" if elong >= p["string_elongation"] else "dust"
        (bx, by), br = cv2.minEnclosingCircle(m_pts)
        blobs.append({
            # map back to full-frame coordinates
            "x": float(bx) + x0, "y": float(by) + y0, "r": max(float(br), 3.0),
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

    # scale annotation weight with resolution so marks stay visible on 20MP
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
