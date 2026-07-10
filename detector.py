"""
detector.py
------------
Per-ROI Z-score anomaly detection for white isolated blobs (dust,
thread, glue - anything bright and white sitting on the module
surface), with PHYSICAL SIZE measurement so the accept/reject rule
matches the actual spec instead of an arbitrary pixel count.

Detection, deliberately simple:
  * LOCAL background detrend (boxFilter) - a broad, slowly-varying
    feature (the coating's own concentric ring) gets absorbed as
    "background" because the local average tracks it closely; a small
    isolated bright speck deviates sharply from its own immediate
    neighbourhood regardless of how much of the ROI a ring covers.
  * One-sided Z-score on the detrended residual (median/MAD, robust)
    - only pixels BRIGHTER than their local surroundings count.
  * White-only gate - coloured pixels (AR-coating reflections) are
    dropped by a plain HSV saturation cutoff.
  * Ring/donut shapes are rejected outright via contour hierarchy (a
    ring has a hole, a real particle doesn't), and anything spanning a
    large fraction of the ROI is rejected as structural, not a defect.
  * No morphological "open" (it erases 1-2px threads); tiny components
    are removed by a pixel-count noise floor instead.
  * Every particle is one class - no shape-based dust/thread split.
    That split was misreading curved bright artefacts (lighting arcs,
    coating structure) as "thread"; a single class removes that
    failure mode. Everything found is a candidate defect, sized and
    gated by the same 0.1 mm spec either way.

Sizing:
  Each blob's size is its MAX FERET LENGTH - the largest end-to-end
  distance across the blob (max pairwise distance between convex-hull
  vertices). For a round particle that is its diameter; for an
  irregular one it is the longest span, matching how the spec is
  written.

Calibration:
  mm_per_px cannot be derived from the image alone - a 20 MP photo says
  nothing about how wide a scene it covers. One physical reference is
  required (see main.py's Settings: two-point measure, optics
  calculation, field of view, or a reference ROI). If mm_per_px is 0
  the detector stays in pixel mode: sizes are reported in px and only
  the pixel noise floor filters blobs.
"""

import cv2
import numpy as np

DUST_COLOR = (0, 0, 255)      # red (BGR) - the only defect colour now

DEFAULT_PARAMS = dict(
    sigma=3.5,                # brightness Z-score threshold
    window_px=0,              # 0 = auto (scales with ROI radius); or set an absolute px value to tune manually
    white_only=True,          # ignore coloured pixels
    white_max_saturation=60,  # HSV S above this = "coloured", not a defect
    dark_floor=0,             # gray below this is ignored (0 = off)
    blur_ksize=3,             # pre-blur to calm sensor noise (odd)
    edge_margin_px=4,         # shrink each ROI so the lens rim doesn't fire
    min_length_mm=0.1,        # SPEC: ignore anything whose max span is under this
    mm_per_px=0.0,            # 0 = uncalibrated (pixel mode)
    min_pixels=10,            # hard noise floor, always applied
    size_bias_px=0.0,         # subtracted from every measured span
    max_defect_fraction=0.5,  # reject any blob spanning more than this fraction of the ROI diameter (structural, not a defect)
    reject_rings=True,        # reject any blob shaped like a ring/donut (has a hole) - coating structure, not a defect
    max_elongation=2.2,       # reject anything more elongated than this (threads, arcs) - only compact round dust survives
)


class Result:
    def __init__(self):
        self.defects = []
        self.verdict = "OK"
        self.annotated = None
        self.calibrated = False

    def largest(self):
        if not self.defects:
            return None
        return max(self.defects, key=lambda b: b["feret_px"])

    def summary(self):
        if not self.defects:
            return "clean"
        n = len(self.defects)
        big = self.largest()
        size = (f"{big['feret_mm']:.3f} mm" if self.calibrated else f"{big['feret_px']:.0f} px")
        return f"{n} defect{'s' if n != 1 else ''}  \u2022  largest {size}"


def _circular_mask(shape_hw, cx, cy, r):
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), max(int(round(r)), 1), 255, -1)
    return m


def _max_feret_px(pts):
    """Largest end-to-end distance across the blob, in pixels.
    The farthest pair of points always lies on the convex hull.
    The +1.0 is a real correction: hull vertices are pixel CENTRES, so a
    blob physically spanning N pixels has centres spanning only N-1."""
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    if len(hull) < 2:
        return 1.0
    d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max())) + 1.0


def _local_mean(values_f32, mask_bool, win):
    """Local neighbourhood average via boxFilter, masked to eligible
    pixels. A broad, slowly-varying feature (a coating ring) is absorbed
    as background since the local mean tracks its level."""
    mask_f = mask_bool.astype(np.float32)
    den = cv2.boxFilter(mask_f, -1, (win, win), normalize=False)
    den = np.maximum(den, 1e-6)
    num = cv2.boxFilter(values_f32 * mask_f, -1, (win, win), normalize=False)
    return num / den


def _robust_scale(residual_2d, mask_bool):
    """A single GLOBAL robust scale (median/MAD) for the whole detrended
    residual population. Using a local (per-pixel) std here instead can
    backfire when a broad feature's own width is comparable to the
    window: the feature's internal gradient then inflates local std
    right on top of it, burying real anomalies in the same area. A
    global scale doesn't have that failure mode - after a reasonable
    local-mean detrend, the residual is near-zero almost everywhere
    (including across the ring), so one robust scale for the whole
    population stays small and a real local spike still stands out."""
    vals = residual_2d[mask_bool]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    std = 1.4826 * mad
    if std < 1.0:
        std = max(float(np.std(vals)), 1.0)
    return med, std


def _bright_mask(gray_c, sat_c, inside, radius, p):
    """The actual candidate-defect mask for one ROI. Order matters:
    1) build the white-eligible mask FIRST - unconditionally black out
       coloured/dark pixels, before any statistics are touched.
    2) compute local mean/std and Z-score using ONLY the white-eligible
       pixels, so a bright/large coloured patch (AR coating) can never
       influence what counts as 'bright' nearby - it's simply excluded
       from the population, not just from the final answer.
    Shared by both real detection and the mask preview, so the preview
    can never show something different from what actually gets detected.
    """
    win = int(p.get("window_px", 0))
    if win <= 0:
        # auto: big enough that any real defect is a small minority of the
        # window's population (so it can't drag its own local mean/std up),
        # but still local enough to absorb a broad ring as background.
        win = max(101, int(round(radius * 0.4)))
    if win % 2 == 0:
        win += 1

    white_ok = inside.copy()
    if p["white_only"] and sat_c is not None:
        white_ok &= (sat_c <= p["white_max_saturation"])
    if p.get("dark_floor", 0) > 0:
        white_ok &= (gray_c >= float(p["dark_floor"]))

    if int(np.count_nonzero(white_ok)) < 50:
        return np.zeros(gray_c.shape, dtype=np.uint8), win

    gray_f = gray_c.astype(np.float32)
    local_mean = _local_mean(gray_f, white_ok, win)
    residual = gray_f - local_mean
    med, local_std = _robust_scale(residual, white_ok)
    z = (residual - med) / local_std
    bright = (z > p["sigma"]) & white_ok

    bright_u8 = bright.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_CLOSE, k)

    # RING REJECTION: a ring/donut shape (the coating's concentric ring)
    # has a HOLE in it - topologically an annulus. A real particle is a
    # solid blob with no hole. This is independent of colour/brightness.
    if p.get("reject_rings", True):
        contours, hierarchy = cv2.findContours(bright_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and len(contours) > 0:
            hierarchy = hierarchy[0]
            for i, cnt in enumerate(contours):
                if hierarchy[i][2] != -1:      # has a child contour = has a hole
                    cv2.drawContours(bright_u8, [cnt], -1, 0, -1)

    return bright_u8, win


def white_mask_preview(frame, rois, params=None):
    """Returns a BGR image showing exactly the real candidate-defect mask:
    everywhere inside each ROI is BLACK except pixels that actually pass
    detection (locally bright, white, not ring-shaped) - not just "not
    coloured". Uses the identical `_bright_mask` as real detection, so
    what you see here is always what gets detected, nothing more."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    else:
        gray = frame
        sat = None
    H, W = gray.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for roi in rois:
        radius = max(roi["r"] - p["edge_margin_px"], 4)
        cx, cy = roi["cx"], roi["cy"]
        pad = 8
        x0 = max(0, int(cx - radius - pad)); x1 = min(W, int(cx + radius + pad) + 1)
        y0 = max(0, int(cy - radius - pad)); y1 = min(H, int(cy + radius + pad) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        gray_c = gray[y0:y1, x0:x1]
        sat_c = sat[y0:y1, x0:x1] if sat is not None else None
        inside = _circular_mask(gray_c.shape, cx - x0, cy - y0, radius) > 0
        if gray_c[inside].size < 50:
            continue
        bright_u8, _ = _bright_mask(gray_c, sat_c, inside, radius, p)
        keep = bright_u8 > 0
        region_out = out[y0:y1, x0:x1]
        region_g3 = g3[y0:y1, x0:x1]
        region_out[keep] = region_g3[keep]
        out[y0:y1, x0:x1] = region_out
    return out


def _detect_in_roi(gray, gray_raw, sat, cx, cy, radius, p):
    H, W = gray.shape
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

    inside = _circular_mask((h, w), cx_c, cy_c, radius) > 0
    if gray_c[inside].size < 50:
        return []

    bright_u8, win = _bright_mask(gray_c, sat_c, inside, radius, p)

    # Sizing mask, measured on the UNBLURRED image (blur inflates small
    # particles by ~2px). Same white-eligibility-first order as detection.
    white_ok = inside.copy()
    if p["white_only"] and sat_c is not None:
        white_ok &= (sat_c <= p["white_max_saturation"])
    if p.get("dark_floor", 0) > 0:
        white_ok &= (gray_raw_c >= float(p["dark_floor"]))
    if int(np.count_nonzero(white_ok)) >= 50:
        gray_raw_f = gray_raw_c.astype(np.float32)
        lm_raw = _local_mean(gray_raw_f, white_ok, win)
        res_raw = gray_raw_f - lm_raw
        med_raw, std_raw = _robust_scale(res_raw, white_ok)
        z_raw = (res_raw - med_raw) / std_raw
        sharp = (z_raw > p["sigma"]) & white_ok
    else:
        sharp = np.zeros_like(inside)

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

        sharp_sel = blob_sel & sharp
        if int(np.count_nonzero(sharp_sel)) >= 2:
            mys, mxs = np.where(sharp_sel)
            m_area = int(sharp_sel.sum())
        else:
            mys, mxs = np.where(blob_sel)
            m_area = area_px
        m_pts = np.column_stack([mxs, mys]).astype(np.int32)

        feret_px = max(_max_feret_px(m_pts) - float(p["size_bias_px"]), 0.1)

        if feret_px > max_feret_allowed:
            continue

        # SHAPE REJECTION: keep only compact, roughly-round blobs (real
        # dust). Anything elongated - a thread, but also a curved
        # lighting arc or coating-structure line - is rejected outright
        # rather than classified. A random arc getting mistaken for a
        # "thread" was a real false positive; simply not treating
        # elongated shapes as defects at all removes that failure mode.
        (_, (rw, rh), _) = cv2.minAreaRect(m_pts)
        elongation = max(rw, rh) / max(min(rw, rh), 1.0)
        if elongation > p.get("max_elongation", 2.2):
            continue

        feret_mm = feret_px * mm_per_px if calibrated else None
        area_mm2 = m_area * (mm_per_px ** 2) if calibrated else None

        if calibrated and feret_mm < p["min_length_mm"]:
            continue

        (bx, by), br = cv2.minEnclosingCircle(m_pts)
        blobs.append({
            "x": float(bx) + x0, "y": float(by) + y0, "r": max(float(br), 3.0),
            "area_px": float(m_area), "feret_px": float(feret_px),
            "feret_mm": feret_mm, "area_mm2": area_mm2,
            "elongation": float(elongation),
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
        cv2.circle(annotated, (int(roi["cx"]), int(roi["cy"])), int(roi["r"]),
                   (90, 90, 96), th)

    for b in res.defects:
        c = (int(b["x"]), int(b["y"]))
        rr = int(b["r"]) + 6 * th
        cv2.circle(annotated, c, rr, DUST_COLOR, th + 1)
        label = f"{b['feret_mm']:.3f}mm" if res.calibrated else f"{b['feret_px']:.0f}px"
        cv2.putText(annotated, label, (c[0] + rr + 4, c[1] - rr),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, DUST_COLOR, th, cv2.LINE_AA)

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
